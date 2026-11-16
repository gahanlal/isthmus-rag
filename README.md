# Isthmus

Isthmus is an experimental document-search tool. It looks for useful passages
in a collection of files, checks nearby text for extra context, and can ask an
AI model to draft an answer with citations.

## Status

This is a research prototype, not a production-ready product. It can miss
important information, and citations do not guarantee that an answer is
correct.

## What we learned from testing

We tested Isthmus on two public data sources. The results are early evidence,
not proof that Isthmus is better than standard search.

On the RepLiQA question set, we compared four ways of choosing text. Each
method saw the same amount of text for a given question. GPT-4o-mini wrote and
scored the answers. Scores are averages from 64 answerable questions that
could be scored across every method. An unanswerable question is one whose
answer is not in the document.

| Method | Accuracy (0–4) | Completeness (0–4) | Supported by sources (0–4) | Correctly said “not in document” |
|---|---:|---:|---:|---:|
| Standard search | 3.30 | **3.45** | **2.89** | 8% |
| Add nearby passages | 3.00 | 3.14 | 2.59 | **21%** |
| Isthmus (fixed-target selection) | 3.30 | 3.38 | 2.56 | 12% |
| Isthmus (marginal-coverage selection) | 3.30 | 3.42 | 2.61 | 0% |

**The main finding:** Isthmus can use less text, but this test did not show that
the smaller, more selective context gives better answers. Standard search had
the strongest completeness and source-support scores. Marginal coverage
usually chose about one document, compared with about 19 for standard search,
but it did not refuse any unanswerable question in this sample.

Scores range from 0 to 4, with higher being better. “Correctly said not in
document” is the share of unanswerable questions where the model did not guess.

This was a 100-question test, and the same AI model wrote and scored answers.
The results need more testing and human review before drawing broad
conclusions.

### Detailed numbers for technical readers

RepLiQA's `repliqa_4` split has 17,950 questions across 3,590 documents. The
full split was indexed into 73,095 chunks. In the 100-question sample, 73
questions were answerable; 68 of those had a reference passage verified in
the source document. The answerability label follows RepLiQA's rule that
`long_answer == "NA"` means unanswerable.

Each document was tagged with its RepLiQA topic as a sector, and all sectors
were kept in one shared search index. Search first looked across the full
index, then used sector scores to select candidate documents; this was not a
test of separate databases for each sector. The full Isthmus pipeline is
represented by the fixed-target and marginal-coverage rows below: both run
global retrieval, sector scoring, graph evidence expansion, and document
selection, with a different selection policy. “Sector + graph” is a component
comparison, not the full pipeline. In this sample, adding sector selection to
graph search did not change the retrieval scores.

“Gold-document hit” means retrieval found an expected source document.
“Exact passage found” and “passage words recovered” measure whether the
annotated answer passage appeared in retrieved text. Context size is the
average whitespace-delimited word and document count.

| Retrieval method | Gold-document hit | Exact passage found | Passage words recovered | Mean context words | Mean documents |
|---|---:|---:|---:|---:|---:|
| Standard search (top-k) | 91.8% | 82.4% | 86.2% | 1,505 | 18.98 |
| Fixed window | 91.8% | **85.3%** | **88.7%** | 3,639 | 18.98 |
| Adaptive graph | 84.9% | 82.4% | 83.0% | 2,746 | 4.98 |
| Sector + graph | 84.9% | 82.4% | 83.0% | 2,746 | 4.98 |
| Isthmus (fixed-target selection) | 84.9% | 83.8% | 84.5% | 1,928 | 3.00 |
| Isthmus (marginal-coverage selection) | 80.8% | 80.9% | 81.5% | **724** | **1.02** |

The separate Docugami test covered 171 questions about 77 PDFs. It matches
retrieved document names to expected names; it does not score answer
correctness.

| Retrieval method | Document recall | Document F1 | Mean context words |
|---|---:|---:|---:|
| Standard search (top-k) | **94.6%** | **47.4%** | 4,255 |
| Fixed document selection | 32.2% | 35.3% | 4,468 |
| Smallest useful context | 32.2% | 35.4% | 1,827 |

## Data sources and reports

- [Docugami KG-RAG-datasets](https://github.com/docugami/KG-RAG-datasets) —
  used for a document-search test on 77 PDFs and 171 questions.
- [ServiceNow RepLiQA](https://huggingface.co/datasets/ServiceNow/repliqa) —
  used for the question-answer test. We indexed the full `repliqa_4` split and
  reviewed a 100-question sample.


Reports:

- [RepLiQA evaluation summary](docs/repliqa-evaluation.md)
- [Full RepLiQA results (JSON)](evaluation/results/repliqa-4-gpt4o-mini-qa.json)
- [RepLiQA answers for human review (CSV)](evaluation/results/repliqa-4-gpt4o-mini-qa.human-review.csv)
- [Docugami evaluation report](docs/docugami-evaluation.md)
- [Docugami passage review](docs/docugami-passage-review.md)

## Run locally

For developers, with Python 3.11 or newer:

```powershell
git clone https://github.com/gahanlal/Contextweave.git
cd Contextweave
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
streamlit run frontend\streamlit_app.py
```

Add any provider keys to a local `.env` file and never commit them. AI provider
requests may incur charges and send document text to that provider. The local
HTTP API has no authentication; do not expose it publicly.

For implementation and evaluation details, see the files linked above and the
[`docs`](docs) folder.
