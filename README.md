# Anvil

Predicts what a product costs from its text description, then uses that model inside a team of agents that scans live online deals and sends a push notification when something is priced well below its estimated value.

Trained on [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) product metadata across 8 categories (Appliances, Automotive, Cell Phones, Electronics, Musical Instruments, Office, Tools, Toys).

## Results

Mean absolute error on **1,000 held-out test products** (lite dataset: 20,000 training items, prices $0.50–$999).

| Model | Error | 95% CI | r² |
|---|---|---|---|
| Constant (training mean) | $127.29 | ±$7.03 | -0.3% |
| XGBoost (bag of words) | $96.43 | ±$6.06 | 34.9% |
| RAG: gpt-4.1-nano + 5 similar products | $84.11 | ±$8.09 | 16.8% |
| **Ensemble: RAG + deep residual network, clamped** | **$77.11** | **±$6.74** | **38.7%** |

The ensemble cuts error **39% vs the baseline** and **20% vs XGBoost**. RAG alone is accurate on most items but makes occasional $900+ overestimates on products outside the training price range; blending with the neural network and clamping to the data's price range removes those, which is why the ensemble's r² is more than double RAG's.

<details>
<summary>All models (first 200 test products)</summary>

| Model | 2 categories (Electronics + Appliances) | 8 categories |
|---|---|---|
| Constant | $209.27 | $132.89 |
| Linear (weight + text length) | $208.85 | $120.61 |
| gpt-4.1-nano zero-shot | $173.65 | $119.79 |
| Deep residual network (289M params, 5 CPU epochs) | $146.75 | $98.04 |
| Bag of words + linear regression | $151.06 | $96.36 |
| Random forest | $147.68 | $96.16 |
| XGBoost | $141.06 | $95.87 |
| 8-layer MLP | $150.40 | $94.21 |
| RAG | $139.76 | $76.58 |
| Ensemble | — | $73.12 |

Every learned model beats the constant baseline by roughly the same ~28% on both datasets; the 2-category set is harder in dollar terms because Electronics and Appliances have a much wider price spread.
</details>

## Architecture

```
Amazon meta (8 × jsonl, ~20 GB) ─► parse ─► dedupe ─► price-weighted sample ─► LLM summaries ─► items
                                                                                      │
                     ┌────────────────────────────────────────────────────────────────┼──────────────────────┐
                     ▼                                                                ▼                      ▼
     baselines · XGBoost · MLP · deep residual network                    Chroma vector store (RAG)    QLoRA (optional, GPU)
                                                                                      │
RSS deal feeds ─► ScannerAgent ─► PlanningAgent ─► EnsembleAgent (RAG 89% · DNN 11%, clamped) ─► MessagingAgent ─► Pushover
```

## What this project adds

Built on the "Price is Right" capstone from Ed Donner's *LLM Engineering* course, rebuilt from notebooks into a tested Python package:

- **Installable package and CLI:** `src/` layout, `anvil` command for every stage, 69 offline tests (APIs and models faked).
- **Resumable data pipeline:** a custom HTTP-range downloader that retries indefinitely and resumes from the exact byte (it completed a 5.35 GB file through 53 dropped connections), plus per-category parse caching.
- **Checkpointed LLM summarization:** every result is written as it arrives, so a crash or Ctrl+C loses nothing; rate limits are retried using the server's suggested wait; runs automatically fall back from OpenAI to Groq when credit runs out. 22,000 products summarized for about $1.70.
- **Ensemble without paid GPU hosting:** the fine-tuned Modal "specialist" is optional (`ANVIL_USE_SPECIALIST`); weights renormalize when it is off, and estimates are clamped to the training price range.
- **Model-aware agents:** reasoning-only API parameters are sent only to reasoning models, so any OpenAI chat model can be dropped in.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -e ".[ml,agents,dev]"                  # add "finetune" for QLoRA on a CUDA GPU
copy .env.example .env                             # add OPENAI_API_KEY, GROQ_API_KEY, HF_TOKEN
```

## Usage

Every command uses the lite dataset by default; add `--full` for the full one.

```bash
# data
anvil curate --per-category 100000 --sample-size 120000 --push   # download, parse, dedupe, sample, split
anvil summarize local --workers 8 --push                         # LLM summaries (resumable)

# models
anvil evaluate constant | linear | nlp-linear | random-forest | xgboost | nn
anvil train-dnn --epochs 5 && anvil evaluate dnn
anvil evaluate frontier
anvil index && anvil evaluate rag
anvil evaluate ensemble --size 1000

# agents
anvil run                  # one pass: scan deals, price them, notify on the best
anvil run --autonomous     # LLM tool-calling planner instead
anvil app                  # Gradio dashboard, re-runs every 5 minutes
```

## Limitations

- Trained on the lite set (20k items); the deep network in particular would benefit from the full dataset and a GPU.
- Prices are capped at $999 by curation, so the system cannot value more expensive products.
- The mix is weighted toward pricier items, and Automotive is down-weighted (5%), following the original sampling strategy.

## Development

```bash
make test     # pytest
make lint     # ruff
```