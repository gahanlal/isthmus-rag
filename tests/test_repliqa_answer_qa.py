"""Test RepLiQA answerability labels, passage metrics, answer QA, and progress persistence."""

import csv
import json
import os

import pytest

from isthmus.models import Chunk
from evaluation.repliqa_answer_qa import (
    _answerability_counts,
    _configure_huggingface_token,
    _correct_report_answerability,
    _is_answerable,
    _looks_like_abstention,
    _passage_metrics,
    _safe_answer_record,
    _write_answer_progress,
    _write_human_review_csv,
    stratified_sample,
    validate_rows,
)


def _row(question_id, topic, answer):
    """Create a RepLiQA fixture row with the supplied question and labels."""
    return {
        "document_id": f"doc-{topic}",
        "document_topic": topic,
        "document_path": f"pdfs/{topic}.pdf",
        "document_extracted": "The answer is in this source passage.",
        "question_id": question_id,
        "question": f"Question {question_id}?",
        "answer": answer,
        "long_answer": (
            "The answer is in this source passage."
            if answer != "UNANSWERABLE"
            else "NA"
        ),
    }


def test_stratified_sample_is_deterministic_and_preserves_topic_answerability():
    """Verify that stratified sample is deterministic and preserves topic answerability."""
    rows = [
        _row(f"{topic}-{answerable}-{number}", topic, "Answer" if answerable else "UNANSWERABLE")
        for topic in ("policy", "news")
        for answerable in (True, False)
        for number in range(2)
    ]

    sample = stratified_sample(rows, 4, seed=7)

    assert sample == stratified_sample(rows, 4, seed=7)
    assert {
        (row["document_topic"], row["answer"] == "UNANSWERABLE")
        for row in sample
    } == {
        ("policy", False),
        ("policy", True),
        ("news", False),
        ("news", True),
    }


def test_stratified_sample_uses_long_answer_na_for_unanswerable_labels():
    """Verify that stratified sample uses long answer na for unanswerable labels."""
    rows = []
    for topic in ("policy", "news"):
        for answerable in (True, False):
            for number in range(10):
                row = _row(f"{topic}-{answerable}-{number}", topic, "Answer")
                if not answerable:
                    row["long_answer"] = "NA"
                rows.append(row)

    sample = stratified_sample(rows, 8, seed=7)

    assert {
        (row["document_topic"], _is_answerable(row))
        for row in sample
    } == {
        ("policy", False),
        ("policy", True),
        ("news", False),
        ("news", True),
    }


def test_huggingface_token_uses_hf_api_without_overwriting_hf_token(monkeypatch):
    """Verify that huggingface token uses hf api without overwriting hf token."""
    monkeypatch.setenv("HF_API", "inference-token")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    _configure_huggingface_token()

    assert os.environ["HF_TOKEN"] == "inference-token"
    monkeypatch.setenv("HF_TOKEN", "explicit-token")
    _configure_huggingface_token()
    assert os.environ["HF_TOKEN"] == "explicit-token"


def test_answerability_counts_use_dataset_rows():
    """Verify that answerability counts use dataset rows."""
    rows = [
        _row("q1", "policy", "Expected answer"),
        _row("q2", "policy", "UNANSWERABLE"),
    ]

    assert _answerability_counts(rows) == {
        "answerable_cases": 1,
        "unanswerable_cases": 1,
    }


def test_long_answer_na_marks_the_replica_question_unanswerable():
    """Verify that long answer na marks the replica question unanswerable."""
    row = _row("q1", "policy", "UNANSWERABLE")
    row["answer"] = "Dataset answer value"
    row["long_answer"] = "NA"

    assert not _is_answerable(row)


def test_report_correction_excludes_invalid_unanswerable_judgments():
    """Verify that report correction excludes invalid unanswerable judgments."""
    row = _row("q1", "policy", "Dataset answer value")
    row["long_answer"] = "NA"
    answer = {
        "text": "The document does not say.",
        "citations": [],
        "cited_evidence": [],
        "citation_format_valid": False,
        "unknown_citations": [],
        "error": None,
    }
    scores = {
        "correctness": 4,
        "completeness": 4,
        "citation_support": 4,
    }
    report = {
        "per_case_retrieval": [{
            "question_id": "q1",
            "answerable": True,
            "arms": {
                "top_k": {
                    "gold_document_hit": True,
                    "source_passage_valid": None,
                    "passage_exact_hit": None,
                    "passage_token_recall": None,
                    "context_words": 1,
                    "context_documents": 1,
                    "latency_ms": 1,
                },
            },
        }],
        "answer_qa": {
            "cases": [{
                "question_id": "q1",
                "document_topic": "policy",
                "answerable": True,
                "answers": {
                    arm: dict(answer)
                    for arm in (
                        "top_k",
                        "adaptive_graph",
                        "fixed_target",
                        "marginal_coverage",
                    )
                },
                "judge_scores": {
                    arm: dict(scores)
                    for arm in (
                        "top_k",
                        "adaptive_graph",
                        "fixed_target",
                        "marginal_coverage",
                    )
                },
                "judge_error": None,
            }],
        },
    }

    _correct_report_answerability(report, [row])

    case = report["answer_qa"]["cases"][0]
    assert case["answerable"] is False
    assert case["expected_answer"] == "UNANSWERABLE"
    assert case["judge_scores"] is None
    assert case["answers"]["top_k"]["citation_format_valid"] is True
    assert report["answer_qa"]["summary"]["unanswerable_cases"] == 1
    assert report["answer_qa"]["summary"]["judged_cases"] == 0
    assert report["retrieval_summary"]["top_k"]["answerable_questions"] == 0


def test_validate_rows_requires_five_questions_per_consistent_document():
    """Verify that validate rows requires five questions per consistent document."""
    rows = [_row(f"q{number}", "policy", "Answer") for number in range(5)]

    assert validate_rows(rows)["doc-policy"]["topic"] == "policy"
    with pytest.raises(ValueError, match="Expected 5 questions per document"):
        validate_rows(rows[:-1])


def test_passage_metrics_recognize_evidence_split_across_chunks():
    """Verify that passage metrics recognize evidence split across chunks."""
    row = _row("q1", "policy", "Answer")
    chunks = (
        Chunk("c1", "doc", "policy", "The answer is in this", 1),
        Chunk("c2", "doc", "policy", "source passage.", 2),
    )

    metrics = _passage_metrics(row, chunks, row["document_extracted"])

    assert metrics == {
        "source_passage_valid": True,
        "passage_exact_hit": True,
        "passage_token_recall": 1.0,
    }


def test_invalid_passage_label_is_not_scored_as_retrieval_miss():
    """Verify that invalid passage label is not scored as retrieval miss."""
    row = _row("q1", "policy", "Answer")

    metrics = _passage_metrics(row, (), "Different source text.")

    assert metrics["source_passage_valid"] is False
    assert metrics["passage_exact_hit"] is None
    assert metrics["passage_token_recall"] is None


def test_unanswerable_reference_and_sanitized_citations():
    """Verify that unanswerable reference and sanitized citations."""
    answer = {
        "text": "The document does not say.",
        "cited_evidence": [{
            "source_id": "S1",
            "document": "source.pdf",
            "location": "line 3",
            "content": "private context text",
        }],
        "citations": [],
        "citation_format_valid": True,
        "unknown_citations": [],
        "error": None,
    }

    assert _looks_like_abstention(answer["text"])
    assert "content" not in _safe_answer_record(answer)["cited_evidence"][0]


def test_human_review_csv_pairs_generated_and_expected_answers(tmp_path):
    """Verify that human review csv pairs generated and expected answers."""
    row = _row("q1", "policy", "Expected answer")
    answer = {
        "text": "Generated answer",
        "citations": ["S1"],
        "cited_evidence": [{"source_id": "S1", "document": "source.pdf", "location": "line 3"}],
        "error": None,
    }
    case = {
        "question_id": "q1",
        "document_topic": "policy",
        "answerable": True,
        "answers": {"top_k": answer},
        "judge_scores": {
            "top_k": {
                "correctness": 4,
                "completeness": 3,
                "citation_support": 4,
            },
        },
        "judge_error": None,
    }
    report_path = tmp_path / "review.csv"

    _write_human_review_csv(report_path, [case], [row], "gpt-4o-mini", "gpt-4o-mini")

    with report_path.open(encoding="utf-8-sig", newline="") as review_file:
        review_rows = list(csv.DictReader(review_file))
    assert len(review_rows) == 1
    assert review_rows[0]["question"] == row["question"]
    assert review_rows[0]["expected_answer"] == "Expected answer"
    assert review_rows[0]["generated_answer"] == "Generated answer"
    assert review_rows[0]["answer_model"] == "gpt-4o-mini"
    assert review_rows[0]["correctness"] == "4"


def test_answer_progress_persists_partial_arm_results(tmp_path):
    """Verify that answer progress persists partial arm results."""
    row = _row("q1", "policy", "Expected answer")
    in_progress = {
        "question_id": "q1",
        "expected_answer": "Expected answer",
        "answers": {"top_k": {"text": "Generated answer"}},
    }
    progress_path = tmp_path / "results" / "answer-progress.json"

    _write_answer_progress(
        progress_path,
        [],
        in_progress,
        [row],
        "gpt-4o-mini",
        "gpt-4o-mini",
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["in_progress"]["answers"]["top_k"]["text"] == "Generated answer"
    assert progress["in_progress"]["expected_answer"] == "Expected answer"
