"""Evaluate Isthmus retrieval and answer quality on the RepLiQA benchmark."""

from __future__ import annotations

import argparse
import csv

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAIError

from evaluation.docugami_answer_qa import _generate_qa_answer, _judge_answers
from evaluation.docugami_benchmark import TokenAwareMiniLM, match_context_words_many
from isthmus.document_parsing import parse_document_bytes
from isthmus.embeddings import embed_many
from isthmus.evaluation.baselines import run_baselines
from isthmus.generation import OpenAIAnswerProvider
from isthmus.index import SQLiteIndex
from isthmus.models import Chunk, Document, Relationship
from isthmus.pipeline import Isthmus, PipelineConfig


DATASET_ID = "ServiceNow/repliqa"
DEFAULT_SPLIT = "repliqa_4"
SPLITS = ("repliqa_0", "repliqa_1", "repliqa_2", "repliqa_3", "repliqa_4")
EXPECTED_QUESTIONS_PER_DOCUMENT = 5
BASELINE_NAMES = {
    "top_k": "A: top-k",
    "fixed_window": "B: fixed window",
    "adaptive_graph": "C: adaptive graph",
    "sector_graph": "D: sector + graph",
    "fixed_target": "E: document reranking",
}
QA_ARMS = ("top_k", "adaptive_graph", "fixed_target", "marginal_coverage")
_TOKEN_PATTERN = re.compile(r"\w+")
_ABSTENTION_PATTERN = re.compile(
    r"\b(?:cannot|can't|unable to|not enough|insufficient|not provided|"
    r"not specified|not stated|unanswerable|does not contain|doesn't contain|does not say)\b",
    re.IGNORECASE,
)


def _is_answerable(row: dict) -> bool:
    answer = str(row["answer"]).strip().casefold()
    long_answer = str(row.get("long_answer", "")).strip().casefold()
    return answer != "unanswerable" and long_answer != "na"


def _answerability_counts(rows: list[dict]) -> dict[str, int]:
    answerable = sum(_is_answerable(row) for row in rows)
    return {
        "answerable_cases": answerable,
        "unanswerable_cases": len(rows) - answerable,
    }


def _configure_huggingface_token() -> None:
    if os.getenv("HF_API") and not os.getenv("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HF_API"]


def _normalized_tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def _stable_rank(seed: int, question_id: str) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()


def stratified_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size < 0:
        raise ValueError("Sample size must not be negative.")
    if sample_size > len(rows):
        raise ValueError(f"Sample size {sample_size} exceeds the {len(rows)} available rows.")
    if sample_size == 0:
        return []
    if sample_size == len(rows):
        return sorted(rows, key=lambda row: row["question_id"])

    strata: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for row in rows:
        strata[(row["document_topic"], _is_answerable(row))].append(row)

    sizes = {key: len(group) for key, group in strata.items()}
    exact_quotas = {
        key: sample_size * size / len(rows)
        for key, size in sizes.items()
    }
    quotas = {key: int(exact) for key, exact in exact_quotas.items()}
    remainder = sample_size - sum(quotas.values())
    remainder_order = sorted(
        strata,
        key=lambda key: (-(exact_quotas[key] - quotas[key]), key),
    )
    for key in remainder_order[:remainder]:
        quotas[key] += 1

    selected = []
    for key, group in strata.items():
        ordered = sorted(
            group,
            key=lambda row: _stable_rank(seed, row["question_id"]),
        )
        selected.extend(ordered[:quotas[key]])
    return sorted(selected, key=lambda row: row["question_id"])


def load_repliqa(split: str) -> tuple[list[dict], str]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            'Install evaluation dependencies with `py -3.14 -m pip install -e ".[evaluation]"`.'
        ) from error

    info = HfApi().dataset_info(DATASET_ID)
    revision = info.sha
    dataset = load_dataset(
        DATASET_ID,
        split=split,
        revision=revision,
    )
    rows = [dict(row) for row in dataset]
    if not rows:
        raise ValueError(f"RepLiQA split {split} contained no rows.")
    return rows, revision


def validate_rows(rows: list[dict]) -> dict[str, dict]:
    required_fields = {
        "document_id",
        "document_topic",
        "document_path",
        "document_extracted",
        "question_id",
        "question",
        "answer",
        "long_answer",
    }
    documents: dict[str, dict] = {}
    question_ids = set()
    questions_per_document: Counter[str] = Counter()
    for row in rows:
        missing = required_fields - row.keys()
        if missing:
            raise ValueError(f"RepLiQA row is missing fields: {sorted(missing)}")
        if row["question_id"] in question_ids:
            raise ValueError(f"Duplicate RepLiQA question id: {row['question_id']}")
        question_ids.add(row["question_id"])
        if not row["question"].strip() or not row["document_extracted"].strip():
            raise ValueError(f"RepLiQA row {row['question_id']} has empty question or document text.")

        document_id = row["document_id"]
        record = {
            "topic": row["document_topic"],
            "path": row["document_path"],
            "text": row["document_extracted"],
        }
        existing = documents.setdefault(document_id, record)
        if existing != record:
            raise ValueError(f"RepLiQA document fields disagree for {document_id}.")
        questions_per_document[document_id] += 1

    invalid_counts = {
        document_id: count
        for document_id, count in questions_per_document.items()
        if count != EXPECTED_QUESTIONS_PER_DOCUMENT
    }
    if invalid_counts:
        raise ValueError(
            f"Expected {EXPECTED_QUESTIONS_PER_DOCUMENT} questions per document; "
            f"found mismatches for {len(invalid_counts)} documents."
        )
    return documents


def _prepare_document(
    split: str,
    source_id: str,
    record: dict,
    provider: TokenAwareMiniLM,
) -> tuple[Document, tuple[Chunk, ...], tuple[Relationship, ...]]:
    document, chunks, relationships = parse_document_bytes(
        record["text"].encode("utf-8"),
        f"{source_id}.txt",
        record["topic"],
        max_words_per_chunk=300,
        embedding_provider=provider,
    )
    index_id = f"repliqa:{split}:{source_id}"
    chunk_ids = {
        chunk.id: f"{index_id}:c{chunk.sequence_number}"
        for chunk in chunks
    }
    document = replace(
        document,
        id=index_id,
        title=Path(record["path"]).stem,
        sector_id=record["topic"],
    )
    chunks = tuple(
        replace(
            chunk,
            id=chunk_ids[chunk.id],
            document_id=index_id,
            sector_id=record["topic"],
        )
        for chunk in chunks
    )
    relationships = tuple(
        replace(
            edge,
            source_chunk_id=chunk_ids[edge.source_chunk_id],
            target_chunk_id=chunk_ids[edge.target_chunk_id],
        )
        for edge in relationships
    )
    return document, chunks, relationships


def index_corpus(
    index: SQLiteIndex,
    split: str,
    records: dict[str, dict],
    provider: TokenAwareMiniLM,
) -> dict[str, str]:
    expected_ids = {
        f"repliqa:{split}:{source_id}": source_id
        for source_id in records
    }
    extras = set(index.documents) - set(expected_ids)
    if extras:
        raise ValueError(
            "The database contains documents outside this RepLiQA split; "
            "use a dedicated database."
        )

    pending = []
    for number, (index_id, source_id) in enumerate(expected_ids.items(), start=1):
        record = records[source_id]
        if index_id in index.documents:
            stored = index.documents[index_id]
            if stored.title != Path(record["path"]).stem or stored.sector_id != record["topic"]:
                raise ValueError(f"Stored RepLiQA metadata differs for {source_id}.")
            continue
        pending.append(_prepare_document(split, source_id, record, provider))
        if number % 250 == 0:
            print(f"Prepared {number}/{len(records)} RepLiQA documents", flush=True)

    flat_chunks = [chunk for _, chunks, _ in pending for chunk in chunks]
    print(f"Embedding {len(flat_chunks)} RepLiQA chunks", flush=True)
    for start in range(0, len(flat_chunks), 64):
        batch = flat_chunks[start:start + 64]
        embeddings = embed_many(provider, [chunk.content for chunk in batch])
        for position, embedding in enumerate(embeddings, start=start):
            flat_chunks[position] = replace(batch[position - start], embedding=embedding)
        if (start + len(batch)) % 640 == 0 or start + len(batch) == len(flat_chunks):
            print(
                f"Embedded {start + len(batch)}/{len(flat_chunks)} RepLiQA chunks",
                flush=True,
            )

    embedded_by_id = {chunk.id: chunk for chunk in flat_chunks}
    for number, (document, chunks, relationships) in enumerate(pending, start=1):
        index.add_document(
            document,
            tuple(embedded_by_id[chunk.id] for chunk in chunks),
            relationships,
        )
        if number % 250 == 0:
            print(f"Indexed {number}/{len(pending)} new RepLiQA documents", flush=True)

    if set(index.documents) != set(expected_ids):
        raise ValueError("The indexed document set does not match the selected RepLiQA split.")
    return {source_id: index_id for index_id, source_id in expected_ids.items()}


def _passage_metrics(
    row: dict,
    retrieved_chunks: tuple[Chunk, ...],
    document_text: str,
) -> dict:
    passage = row["long_answer"].strip()
    if not _is_answerable(row) or passage.casefold() == "na":
        return {
            "source_passage_valid": None,
            "passage_exact_hit": None,
            "passage_token_recall": None,
        }

    reference = _normalized_tokens(passage)
    source_tokens = _normalized_tokens(document_text)
    source_valid = bool(reference) and _contains_tokens(source_tokens, reference)
    if not source_valid:
        return {
            "source_passage_valid": False,
            "passage_exact_hit": None,
            "passage_token_recall": None,
        }

    retrieved_text = " ".join(
        chunk.content
        for chunk in sorted(retrieved_chunks, key=lambda chunk: (
            chunk.document_id,
            chunk.sequence_number,
        ))
    )
    retrieved = _normalized_tokens(retrieved_text)
    retrieved_counts = Counter(retrieved)
    token_recall = sum(
        min(count, retrieved_counts[token])
        for token, count in Counter(reference).items()
    ) / len(reference)
    return {
        "source_passage_valid": True,
        "passage_exact_hit": _contains_tokens(retrieved, reference),
        "passage_token_recall": token_recall,
    }


def _contains_tokens(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[start:start + width] == needle for start in range(len(haystack) - width + 1))


def _retrieval_summary(per_case: list[dict], arm_names: tuple[str, ...]) -> dict:
    answerable = [case for case in per_case if case["answerable"]]
    summary = {}
    for arm in arm_names:
        rows = [case["arms"][arm] for case in per_case]
        valid_passages = [
            row for row in answerable
            if row["arms"][arm]["source_passage_valid"] is True
        ]
        summary[arm] = {
            "questions": len(rows),
            "answerable_questions": len(answerable),
            "gold_document_hit_rate": statistics.mean(
                row["arms"][arm]["gold_document_hit"] for row in answerable
            ) if answerable else None,
            "source_passage_valid_rate": (
                sum(row["arms"][arm]["source_passage_valid"] is True for row in answerable)
                / len(answerable)
            ) if answerable else None,
            "passage_exact_hit_rate_on_valid_labels": statistics.mean(
                row["arms"][arm]["passage_exact_hit"] for row in valid_passages
            ) if valid_passages else None,
            "mean_passage_token_recall_on_valid_labels": statistics.mean(
                row["arms"][arm]["passage_token_recall"] for row in valid_passages
            ) if valid_passages else None,
            "mean_context_words": statistics.mean(row["context_words"] for row in rows) if rows else None,
            "mean_context_documents": statistics.mean(row["context_documents"] for row in rows) if rows else None,
            "median_latency_ms": statistics.median(row["latency_ms"] for row in rows) if rows else None,
        }
    return summary


def _correct_report_answerability(report: dict, rows: list[dict]) -> dict:
    rows_by_id = {row["question_id"]: row for row in rows}
    for case in report["per_case_retrieval"]:
        case["answerable"] = _is_answerable(rows_by_id[case["question_id"]])
    report["retrieval_summary"] = _retrieval_summary(
        report["per_case_retrieval"],
        tuple(report["per_case_retrieval"][0]["arms"]),
    )
    answerable_rows = [
        rows_by_id[case["question_id"]]
        for case in report["per_case_retrieval"]
        if case["answerable"]
    ]
    report["valid_gold_passage_rate"] = (
        sum(
            _passage_metrics(row, (), row["document_extracted"])[
                "source_passage_valid"
            ] is True
            for row in answerable_rows
        ) / len(answerable_rows)
        if answerable_rows else None
    )
    for case in report["answer_qa"]["cases"]:
        row = rows_by_id[case["question_id"]]
        answerable = _is_answerable(row)
        case["question"] = row["question"]
        case["answerable"] = answerable
        case["expected_answer"] = row["answer"] if answerable else "UNANSWERABLE"
        case["reference_passage"] = (
            row["long_answer"]
            if answerable and row["long_answer"].strip().casefold() != "na"
            else None
        )
        if not answerable:
            case["judge_scores"] = None
            case["judge_error"] = {
                "type": "excluded_after_label_correction",
                "message": "The previous judge score used an answerable reference and was excluded.",
            }
        for answer in case["answers"].values():
            if (
                not answerable
                and answer["error"] is None
                and _looks_like_abstention(answer["text"])
                and not answer["unknown_citations"]
            ):
                answer["citation_format_valid"] = True
    report["answerability_rule"] = (
        "A row is unanswerable when answer is UNANSWERABLE or long_answer is NA; "
        "the RepLiQA dataset card states long_answer is NA iff the answer is UNANSWERABLE."
    )
    report["answer_qa"]["summary"] = _answer_qa_summary(
        report["answer_qa"]["cases"]
    )
    report["answer_qa"]["label_correction_note"] = (
        "Unanswerable cases were reclassified from the source long_answer marker. "
        "Their previous judge scores were excluded; generated answers were retained "
        "and are scored for abstention separately."
    )
    return report


def _looks_like_abstention(text: str) -> bool:
    return bool(_ABSTENTION_PATTERN.search(text))


def _safe_answer_record(answer: dict) -> dict:
    result = {
        key: value
        for key, value in answer.items()
        if key not in ("cited_evidence",)
    }
    result["cited_evidence"] = [
        {key: citation[key] for key in ("source_id", "document", "location")}
        for citation in answer["cited_evidence"]
    ]
    return result


def _write_human_review_csv(
    path: Path,
    cases: list[dict],
    rows: list[dict],
    answer_model: str,
    judge_model: str,
) -> None:
    rows_by_id = {row["question_id"]: row for row in rows}
    fieldnames = (
        "question_id",
        "document_topic",
        "answerable",
        "question",
        "expected_answer",
        "reference_passage",
        "answer_model",
        "arm",
        "generated_answer",
        "citations",
        "cited_evidence",
        "correctness",
        "completeness",
        "citation_support",
        "answer_error",
        "judge_error",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            row = rows_by_id[case["question_id"]]
            answerable = _is_answerable(row)
            for arm, answer in case["answers"].items():
                scores = (case["judge_scores"] or {}).get(arm, {})
                writer.writerow({
                    "question_id": case["question_id"],
                    "document_topic": case["document_topic"],
                    "answerable": answerable,
                    "question": row["question"],
                    "expected_answer": (
                        row["answer"] if answerable else "UNANSWERABLE"
                    ),
                    "reference_passage": (
                        row["long_answer"]
                        if answerable and row["long_answer"].strip().casefold() != "na"
                        else ""
                    ),
                    "answer_model": answer_model,
                    "arm": arm,
                    "generated_answer": answer["text"] or "",
                    "citations": json.dumps(answer["citations"], ensure_ascii=False),
                    "cited_evidence": json.dumps(
                        answer["cited_evidence"], ensure_ascii=False
                    ),
                    "correctness": scores.get("correctness"),
                    "completeness": scores.get("completeness"),
                    "citation_support": scores.get("citation_support"),
                    "answer_error": json.dumps(answer["error"], ensure_ascii=False),
                    "judge_error": json.dumps(
                        case["judge_error"], ensure_ascii=False
                    ),
                })


def _write_answer_progress(
    path: Path,
    cases: list[dict],
    in_progress: dict | None,
    rows: list[dict],
    answer_model: str,
    judge_model: str,
) -> None:
    payload = {
        "question_ids": [row["question_id"] for row in rows],
        "answer_model": answer_model,
        "judge_model": judge_model,
        "cases": cases,
        "in_progress": in_progress,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _answer_qa(
    rows: list[dict],
    answer_bundles: dict[str, dict],
    index: SQLiteIndex,
    pipelines: dict[str, Isthmus],
    answer_model: str,
    judge_model: str,
    progress_path: Path | None = None,
) -> tuple[list[dict], dict]:
    provider = OpenAIAnswerProvider(answer_model, os.getenv("ISTHMUS_ANSWER_BASE_URL") or None)
    question_ids = [row["question_id"] for row in rows]
    cases = []
    in_progress = None
    if progress_path is not None and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress["question_ids"] != question_ids
            or progress["answer_model"] != answer_model
            or progress["judge_model"] != judge_model
        ):
            raise ValueError(
                f"Answer progress at {progress_path} does not match this run."
            )
        cases = progress["cases"]
        in_progress = progress.get("in_progress")
    completed_ids = {case["question_id"] for case in cases}
    pending_ids = [question_id for question_id in question_ids if question_id not in completed_ids]
    if in_progress is not None and (
        not pending_ids or in_progress["question_id"] != pending_ids[0]
    ):
        raise ValueError(f"Answer progress at {progress_path} has an unexpected in-progress case.")
    for number, row in enumerate(rows, start=1):
        if row["question_id"] in completed_ids:
            continue
        bundle = answer_bundles[row["question_id"]]
        base_result = bundle["base_result"]
        matched = match_context_words_many(
            bundle["contexts"],
            pipelines["fixed_target"].config.global_token_budget,
        )
        context_words = {
            arm: sum(chunk.token_count for chunk in chunks)
            for arm, chunks in matched.items()
        }
        if len(set(context_words.values())) != 1:
            raise ValueError(f"QA context sizes differ for {row['question_id']}.")
        answerable = _is_answerable(row)
        if in_progress is None:
            in_progress = {
                "question_id": row["question_id"],
                "document_topic": row["document_topic"],
                "question": row["question"],
                "answerable": answerable,
                "expected_answer": row["answer"] if answerable else "UNANSWERABLE",
                "answers": {},
            }
            if progress_path is not None:
                _write_answer_progress(
                    progress_path,
                    cases,
                    in_progress,
                    rows,
                    answer_model,
                    judge_model,
                )
        answers = in_progress["answers"]
        for arm in QA_ARMS:
            if arm in answers:
                continue
            arm_result = replace(
                base_result,
                context=matched[arm],
                evidence_coverage={},
            )
            try:
                answer = _generate_qa_answer(arm_result, index.documents, provider)
                if (
                    not answerable
                    and _looks_like_abstention(answer["text"])
                    and not answer["unknown_citations"]
                ):
                    answer["citation_format_valid"] = True
                answers[arm] = answer
            except (OpenAIError, ValueError) as error:
                answers[arm] = {
                    "text": None,
                    "cited_evidence": [],
                    "citations": [],
                    "citation_format_valid": False,
                    "unknown_citations": [],
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            if progress_path is not None:
                _write_answer_progress(
                    progress_path,
                    cases,
                    in_progress,
                    rows,
                    answer_model,
                    judge_model,
                )

        order = list(QA_ARMS)
        order.sort(key=lambda arm: _stable_rank(int(row["question_id"].encode().hex()[:8], 16), arm))
        blind_labels = {
            arm: chr(ord("A") + position)
            for position, arm in enumerate(order)
        }
        reference_passages = []
        if answerable and row["long_answer"].strip().casefold() != "na":
            reference_passages.append({
                "document_path": row["document_path"],
                "text": row["long_answer"],
            })
        required_facts = (
            [f"Reference answer: {row['answer']}"]
            if answerable
            else ["The answer is not present in the reference document; abstain rather than guess."]
        )
        judgment = None
        judge_error = None
        if all(answer["error"] is None for answer in answers.values()):
            blinded_answers = {
                blind_labels[arm]: {
                    "answer": answers[arm]["text"],
                    "cited_retrieved_passages": answers[arm]["cited_evidence"],
                }
                for arm in QA_ARMS
            }
            try:
                raw_judgment = _judge_answers(
                    provider,
                    judge_model,
                    row["question"],
                    required_facts,
                    reference_passages,
                    blinded_answers,
                )
                judgment = {
                    arm: raw_judgment[blind_labels[arm]]
                    for arm in QA_ARMS
                }
            except (OpenAIError, ValueError) as error:
                judge_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }

        cases.append({
            "question_id": row["question_id"],
            "document_topic": row["document_topic"],
            "question": row["question"],
            "answerable": answerable,
            "expected_answer": row["answer"] if answerable else "UNANSWERABLE",
            "reference_passage": (
                row["long_answer"]
                if answerable and row["long_answer"].strip().casefold() != "na"
                else None
            ),
            "context_words_per_arm": next(iter(context_words.values())),
            "blind_labels": blind_labels,
            "answers": {
                arm: _safe_answer_record(answers[arm])
                for arm in QA_ARMS
            },
            "judge_scores": judgment,
            "judge_error": judge_error,
        })
        if progress_path is not None:
            _write_answer_progress(
                progress_path,
                cases,
                None,
                rows,
                answer_model,
                judge_model,
            )
        in_progress = None
        completed_ids.add(row["question_id"])
        print(f"Evaluated RepLiQA answer {number}/{len(rows)}", flush=True)

    return cases, _answer_qa_summary(cases)


def _answer_qa_summary(cases: list[dict]) -> dict:
    judged = [case for case in cases if case["judge_scores"] is not None]
    summary = {
        "requested_cases": len(cases),
        "judged_cases": len(judged),
        "answerable_cases": sum(case["answerable"] for case in cases),
        "unanswerable_cases": sum(not case["answerable"] for case in cases),
        "arms": {},
        "pairwise_score_outcomes": {},
    }
    for arm in QA_ARMS:
        scored = [case["judge_scores"][arm] for case in judged]
        unanswerable_answers = [
            case["answers"][arm]["text"]
            for case in cases
            if not case["answerable"] and case["answers"][arm]["error"] is None
        ]
        summary["arms"][arm] = {
            key: statistics.mean(score[key] for score in scored) if scored else None
            for key in ("correctness", "completeness", "citation_support")
        }
        summary["arms"][arm]["answer_failures"] = sum(
            case["answers"][arm]["error"] is not None for case in cases
        )
        summary["arms"][arm]["citation_format_failures"] = sum(
            case["answers"][arm]["error"] is None
            and not case["answers"][arm]["citation_format_valid"]
            for case in cases
        )
        summary["arms"][arm]["unanswerable_abstention_rate"] = (
            statistics.mean(_looks_like_abstention(text) for text in unanswerable_answers)
            if unanswerable_answers else None
        )
    for left_index, left in enumerate(QA_ARMS):
        for right in QA_ARMS[left_index + 1:]:
            wins = ties = losses = 0
            for case in judged:
                left_score = sum(case["judge_scores"][left][key] for key in (
                    "correctness", "completeness", "citation_support",
                ))
                right_score = sum(case["judge_scores"][right][key] for key in (
                    "correctness", "completeness", "citation_support",
                ))
                if left_score > right_score:
                    wins += 1
                elif right_score > left_score:
                    losses += 1
                else:
                    ties += 1
            summary["pairwise_score_outcomes"][f"{left}_vs_{right}"] = {
                f"{left}_wins": wins,
                f"{right}_wins": losses,
                "ties": ties,
            }
    return summary


def main() -> None:
    load_dotenv()
    _configure_huggingface_token()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=SPLITS, default=DEFAULT_SPLIT)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--retrieval-limit",
        type=int,
        default=0,
        help="Evaluate a deterministic stratified subset; 0 evaluates the entire split.",
    )
    parser.add_argument(
        "--answer-sample-size",
        type=int,
        default=100,
        help="Number of stratified cases for LLM answer QA; 0 disables answer generation.",
    )
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument(
        "--execution-provider",
        choices=("cpu", "directml", "cuda"),
        default="cpu",
        help="ONNX Runtime inference provider for MiniLM embeddings.",
    )
    parser.add_argument("--answer-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default=os.getenv("ISTHMUS_QA_JUDGE_MODEL") or "gpt-4o-mini")
    args = parser.parse_args()
    if args.retrieval_limit < 0 or args.answer_sample_size < 0:
        parser.error("Sample sizes must not be negative.")

    rows, dataset_revision = load_repliqa(args.split)
    records = validate_rows(rows)
    if args.retrieval_limit:
        retrieval_rows = stratified_sample(rows, args.retrieval_limit, args.seed)
    else:
        retrieval_rows = sorted(rows, key=lambda row: row["question_id"])
    answer_rows = stratified_sample(
        retrieval_rows,
        min(args.answer_sample_size, len(retrieval_rows)),
        args.seed + 1,
    )
    answer_ids = {row["question_id"] for row in answer_rows}
    print(
        f"Loaded {len(rows)} RepLiQA rows across {len(records)} documents "
        f"from {args.split} at {dataset_revision}.",
        flush=True,
    )

    provider_names = {
        "cpu": ["CPUExecutionProvider"],
        "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    }
    preferred_providers = provider_names[args.execution_provider]
    provider = TokenAwareMiniLM(preferred_providers=preferred_providers)
    available_providers = provider.model.ort.get_available_providers()
    if preferred_providers[0] not in available_providers:
        parser.error(
            f"Requested {preferred_providers[0]}, but available ONNX providers are "
            f"{available_providers}."
        )
    provider.provider_id = f"{provider.provider_id}:{args.execution_provider}"
    configs = {
        policy: PipelineConfig(document_selection_policy=policy)
        for policy in ("fixed_target", "marginal_coverage")
    }
    retrieval_started = time.perf_counter()
    answer_bundles = {}
    per_case = []
    document_index_ids = {
        source_id: f"repliqa:{args.split}:{source_id}"
        for source_id in records
    }
    with SQLiteIndex(provider, args.database) as index:
        document_index_ids = index_corpus(index, args.split, records, provider)
        pipelines = {
            policy: Isthmus(index, config)
            for policy, config in configs.items()
        }
        for number, row in enumerate(retrieval_rows, start=1):
            baseline_sets = {
                policy: {
                    baseline.name: baseline
                    for baseline in run_baselines(
                        index,
                        row["question"],
                        configs[policy],
                        pipelines[policy],
                    )
                }
                for policy in configs
            }
            reference_policy = baseline_sets["fixed_target"]
            marginal_policy = baseline_sets["marginal_coverage"]
            for name in (
                BASELINE_NAMES["top_k"],
                BASELINE_NAMES["fixed_window"],
                BASELINE_NAMES["adaptive_graph"],
                BASELINE_NAMES["sector_graph"],
            ):
                if tuple(
                    chunk.id for chunk in reference_policy[name].chunks
                ) != tuple(
                    chunk.id for chunk in marginal_policy[name].chunks
                ):
                    raise ValueError(f"Baseline {name} changed with selection policy.")

            arm_results = {
                **{
                    arm: reference_policy[name]
                    for arm, name in BASELINE_NAMES.items()
                },
                "marginal_coverage": marginal_policy[BASELINE_NAMES["fixed_target"]],
            }
            answerable = _is_answerable(row)
            gold_index_id = document_index_ids[row["document_id"]]
            arm_metrics = {}
            for arm, result in arm_results.items():
                actual_document_ids = set(result.document_ids)
                gold_chunks = tuple(
                    chunk for chunk in result.chunks
                    if chunk.document_id == gold_index_id
                )
                passage = _passage_metrics(
                    row,
                    gold_chunks,
                    records[row["document_id"]]["text"],
                )
                arm_metrics[arm] = {
                    "gold_document_hit": (
                        float(gold_index_id in actual_document_ids)
                        if answerable else None
                    ),
                    **passage,
                    "chunks": len(result.chunks),
                    "context_words": sum(chunk.token_count for chunk in result.chunks),
                    "context_documents": len(result.document_ids),
                    "latency_ms": result.latency_ms,
                }

            per_case.append({
                "question_id": row["question_id"],
                "document_topic": row["document_topic"],
                "answerable": answerable,
                "arms": arm_metrics,
            })
            if row["question_id"] in answer_ids:
                qa_pipeline_result = pipelines["fixed_target"].query(row["question"])
                contexts = {
                    arm: tuple(result.chunks)
                    for arm, result in (
                        ("top_k", arm_results["top_k"]),
                        ("adaptive_graph", arm_results["adaptive_graph"]),
                        ("fixed_target", arm_results["fixed_target"]),
                        ("marginal_coverage", arm_results["marginal_coverage"]),
                    )
                }
                answer_bundles[row["question_id"]] = {
                    "contexts": contexts,
                    "base_result": qa_pipeline_result,
                }
            if number % 100 == 0:
                print(f"Evaluated retrieval {number}/{len(retrieval_rows)}", flush=True)

        retrieval_summary = _retrieval_summary(
            per_case,
            tuple(arm_results),
        )
        answer_cases = []
        answer_summary = {
            "requested_cases": 0,
            "judged_cases": 0,
            "answerable_cases": 0,
            "unanswerable_cases": 0,
            "arms": {},
            "pairwise_score_outcomes": {},
        }
        if answer_rows:
            answer_cases, answer_summary = _answer_qa(
                answer_rows,
                answer_bundles,
                index,
                pipelines,
                args.answer_model,
                args.judge_model,
                args.report.with_suffix(".qa-progress.json"),
            )

        report = {
            "dataset": DATASET_ID,
            "dataset_license": "CC BY 4.0",
            "dataset_revision": dataset_revision,
            "split": args.split,
            "split_recommendation": "Latest released split, as recommended by the RepLiQA dataset card.",
            "embedding_model": provider.provider_id,
            "inference_providers": provider.model.model.get_providers(),
            "retrieval_configuration": {
                "global_top_k": configs["fixed_target"].global_top_k,
                "global_word_budget": configs["fixed_target"].global_token_budget,
                "chunking": "Project text parser, 300-word maximum with the embedding model's token limit.",
                "document_selection_policies": ["fixed_target", "marginal_coverage"],
            },
            "retrieval_questions": len(retrieval_rows),
            "retrieval_sample_seed": args.seed if args.retrieval_limit else None,
            "dataset_questions": len(rows),
            "indexed_documents": len(index.documents),
            "indexed_chunks": len(index.chunks),
            "valid_gold_passage_rate": (
                sum(
                    _passage_metrics(row, (), records[row["document_id"]]["text"])[
                        "source_passage_valid"
                    ] is True
                    for row in retrieval_rows
                    if _is_answerable(row)
                )
                / sum(_is_answerable(row) for row in retrieval_rows)
                if any(_is_answerable(row) for row in retrieval_rows)
                else None
            ),
            "retrieval_summary": retrieval_summary,
            "answer_qa": {
                "answer_model": args.answer_model if answer_rows else None,
                "judge_model": args.judge_model if answer_rows else None,
                "human_review_file": (
                    str(args.report.with_suffix(".human-review.csv"))
                    if answer_rows else None
                ),
                "sample_seed": (
                    args.seed + 1
                    if 0 < args.answer_sample_size < len(retrieval_rows)
                    else None
                ),
                "sample_strategy": (
                    "Deterministic stratification by document topic and answerability; "
                    "when every retrieval question is used, no second QA sample is drawn."
                ),
                "context_matching": (
                    "For each QA case, top-k, adaptive graph, fixed-target, "
                    "and marginal-coverage contexts are truncated to the exact "
                    "same whitespace-word count."
                ),
                "sample_size": args.answer_sample_size,
                "summary": answer_summary,
                "cases": answer_cases,
            },
            "total_retrieval_and_qa_seconds": time.perf_counter() - retrieval_started,
            "per_case_retrieval": per_case,
        }
    if answer_rows:
        _write_human_review_csv(
            args.report.with_suffix(".human-review.csv"),
            answer_cases,
            answer_rows,
            args.answer_model,
            args.judge_model,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    progress_path = args.report.with_suffix(".qa-progress.json")
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps({
        "retrieval_summary": report["retrieval_summary"],
        "answer_qa_summary": report["answer_qa"]["summary"],
        "report": str(args.report),
        "human_review_file": report["answer_qa"]["human_review_file"],
    }, indent=2))


if __name__ == "__main__":
    main()
