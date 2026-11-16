# RepLiQA answer-quality evaluation

## Scope and setup

This is a 100-question, topic-stratified evaluation using the
[ServiceNow RepLiQA dataset](https://github.com/ServiceNow/repliqa), split
`repliqa_4`, revision `bc880adc948fd3a70d5f8b2b3a1d1ee90d820dbd`. The full
split (17,950 QA rows across 3,590 documents) was indexed into 73,095
token-aware chunks. MiniLM embeddings ran through ONNX Runtime DirectML; answer
generation and judging used GPT-4o-mini.

The sample contains 73 answerable and 27 unanswerable questions. RepLiQA
defines `long_answer == "NA"` iff a question is unanswerable. The evaluator
now uses that marker as well as `answer == "UNANSWERABLE"`; the initial run
missed the former marker. Generated answers were retained, but judge scores
from the incorrectly classified unanswerable cases were excluded. Unanswerable
performance is reported as abstention rate instead.

Each document's RepLiQA `document_topic` was copied to the document and chunk
sector IDs. All topics stayed in one shared index: retrieval searched globally
first, then scored sectors to select candidates. The sector + graph arm and
adaptive graph had identical retrieval metrics in this sample, so sector
labeling did not show an added retrieval benefit here.

The full `Isthmus.query` pipeline is represented by the fixed-target and
marginal-coverage arms. Both perform global retrieval, sector scoring,
graph-based evidence expansion, and document selection; they differ in the
document-selection policy. The sector + graph arm is a component comparison,
not the complete pipeline.

For each question, top-k, adaptive graph, fixed-target and marginal-coverage
contexts were clipped to the same whitespace-word count. This was the smallest
available context for that question, not a fixed count across all questions.
The same GPT-4o-mini model generated and judged answers, so its scores are an
automated comparison rather than independent human evaluation. The report
records every question and arm; the CSV is formatted for side-by-side human
review.

## Retrieval results

Gold-document hit rate and passage metrics below are computed on the 73
answerable cases. Passage metrics use only the 68 source `long_answer` labels
that were verified to occur in the extracted source document.

| Retrieval arm | Gold-document hit | Passage exact hit | Passage-token recall | Mean natural context words | Mean documents |
|---|---:|---:|---:|---:|---:|
| Top-k | 91.8% | 82.4% | 86.2% | 1,505 | 18.98 |
| Fixed window | 91.8% | **85.3%** | **88.7%** | 3,639 | 18.98 |
| Adaptive graph | 84.9% | 82.4% | 83.0% | 2,746 | 4.98 |
| Sector + graph | 84.9% | 82.4% | 83.0% | 2,746 | 4.98 |
| Isthmus — fixed target | 84.9% | 83.8% | 84.5% | 1,928 | 3.00 |
| Isthmus — marginal coverage | 80.8% | 80.9% | 81.5% | **724** | **1.02** |

Marginal coverage returned the smallest context, but also had the lowest
gold-document hit rate and passage recall in this sample. Fixed target reduced
the number of selected documents substantially relative to top-k, with a
smaller decline in gold-document hits. These are retrieval indicators, not
proof that a retrieved passage entails an answer.

## Answer-quality results

GPT-4o-mini judged 64 answerable questions for which all four arms had
successful answers. Scores are means on the judge's 0–4 scale. The 27
unanswerable cases are excluded from these scores; their abstention rate is
measured separately from generated text.

| Arm | Correctness | Completeness | Citation support | Unanswerable abstention | Answer failures | Citation-format failures |
|---|---:|---:|---:|---:|---:|---:|
| Top-k | **3.30** | **3.45** | **2.89** | 8.0% | 3 | 14 |
| Adaptive graph | 3.00 | 3.14 | 2.59 | **20.8%** | 4 | 20 |
| Isthmus — fixed target | **3.30** | 3.38 | 2.56 | 11.5% | **1** | 21 |
| Isthmus — marginal coverage | **3.30** | 3.42 | 2.61 | 0.0% | 2 | 25 |

There is no clear overall answer-quality win for the selective arms here:
correctness is tied, while top-k has the highest mean completeness and
citation-support scores. The low abstention rates are a notable weakness,
especially for marginal coverage. The results do not establish that
selectivity improves answer quality; they suggest a context-size versus
evidence-coverage trade-off that needs more cases and human review.

## Reproduction and artifacts

The runner uses `HF_API` for Hugging Face authentication (mapped to
`HF_TOKEN`) and `OPENAI_API_KEY` for GPT-4o-mini. DirectML requires
`onnxruntime-directml` in the selected Python environment.

```powershell
py -3.14 -m evaluation.repliqa_answer_qa `
  --execution-provider directml `
  --split repliqa_4 `
  --database evaluation/results/repliqa-4.sqlite `
  --report evaluation/results/repliqa-4-gpt4o-mini-qa.json `
  --retrieval-limit 100 `
  --answer-sample-size 100 `
  --answer-model gpt-4o-mini `
  --judge-model gpt-4o-mini `
  --seed 20261116
```

- [Full JSON report](../evaluation/results/repliqa-4-gpt4o-mini-qa.json)
- [Human-review CSV](../evaluation/results/repliqa-4-gpt4o-mini-qa.human-review.csv)

The dataset is licensed CC BY 4.0. Its copied `long_answer` annotations are
validated against the extracted source before passage-retrieval scoring; the
dataset card cautions that these copied passages are not independently checked
for source presence.
