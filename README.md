# Financial Agent Safety Bench

An adversarial robustness benchmark for tool-using financial research agents. It tests whether an agent that searches the web, reads SEC filings, and pulls price data stays accurate and safe when its inputs are wrong, misleading, or hostile.

The benchmark includes a reference agent, a tool-level attack environment, 374 test cases, and an evaluation pipeline that scores each run with an LLM judge and deterministic checks.

## Attack taxonomy

Attacks are grouped by what the attacker wants and where the attack enters. Strength, tone, and style are variants within an attack.

| Class | Attack | Entry point | Variants | Cases |
|---|---|---|---|---|
| False figures | M1 False figure from the user | User | tentative, assertive, authority | 24 |
| | M2 False figure in search results | Search | single, corroborated, conflicting | 24 |
| | M3 Tampered filing | Filing page | ×10, one filing, all filings | 24 |
| | M4 Fabricated undisclosed metric | Search | one | 12 |
| False premises | P1 Absurd event | User | open, anchored | 24 |
| | P2 Plausible false event | User | invented company, real company | 18 |
| | P3 Overstated cause | User | one | 12 |
| User pressure | U1 Forced conclusion | User | one | 12 |
| | U2 One-sided evidence | User | one | 12 |
| Hidden instructions | I1 Add text to the report | Search, filing page | 6 styles | 22 |
| | I2 Add an external link | Search, filing page | 6 styles | 22 |
| | I3 Force an investment rating | Search, filing page | 6 styles | 22 |
| | I4 Make the agent open a URL | Search, filing page | 6 styles | 22 |
| Confidential information | C1 Secret not needed by the task | User | explicit, implicit | 24 |
| | C2 Secret relevant to the task | User | one | 12 |
| | C3 Task requires looking up the client | User | one | 12 |
| Baselines | Clean runs of the same tasks | None | | 76 |

The hidden-instruction styles are naive, context ignoring, fake completion (Liu et al., USENIX Security 2024), important message (AgentDojo, NeurIPS 2024), house style, and an instruction addressed to the retrieval sub-model.

## What is in this repository

The public case file, `safety_bench/dataset/cases.jsonl`, holds 198 of the 374 cases. The other 176 are built on [FrontierFinance](https://huggingface.co/datasets/samaya-ai/FrontierFinance) questions and expert rubrics. That dataset is gated, and its card asks users not to reshare it outside a gated or private repository, so this repository contains no FrontierFinance text. Those cases are rebuilt locally from your own copy:

| Cases | Attacks | Needs FrontierFinance |
|---|---|---|
| 198 in the repository | M4, P1–P3, U1–U2, C1–C3, and their baselines | No |
| 176 built locally | M1–M3, I1–I4, and their baselines | Yes |

The build checks that it reproduces the frozen case set exactly, and that no FrontierFinance text reaches the public file. Where each case comes from is documented in [safety_bench/dataset/README.md](safety_bench/dataset/README.md).

### About the attack data

The cases contain attack material by design: false financial figures attributed to real companies, invented companies, clients and code names, fabricated news sources and web addresses, and hidden instructions aimed at AI agents. All of it is synthetic and exists to test agents. None of it is a statement of fact. Do not use it as financial information or as training data for models meant to report facts. The invented web domains are not owned by this project.

## Setup

Requires Python 3.11 or later.

```bash
uv sync                 # or: python -m venv .venv && pip install -e .
cp .env.example .env    # then fill in the keys
```

| Variable | Used for |
|---|---|
| `OPENAI_API_KEY` | The reference agent and the default judge |
| `TAVILY_API_KEY` | Web search |
| `PRICING_DATA_API_KEY` | Price history (Tiingo) |
| `SEC_USER_AGENT` | SEC EDGAR access, which requires your name and email |
| `FRONTIER_FINANCE_DATA` | Optional path to `frontier_finance_public.jsonl` |

To use the full case set, request access to FrontierFinance on Hugging Face, download `frontier_finance_public.jsonl`, and place it in `data/` or point `FRONTIER_FINANCE_DATA` at it.

## Usage

```bash
python safety_bench/build/build_dataset.py             # public cases; also the full set if the source data is present
python safety_bench/build/validate_dataset.py          # add --online to check invented names and domains
python -m pytest tests

python safety_bench/run.py --tag demo --preview 3 --workers 6    # a fixed sample: the first 3 cases of each variant
python safety_bench/run.py --tag demo --ids C3_wmt_default,P1_ko_open
python safety_bench/run.py --tag main --stage 1 --workers 6      # every case once
python safety_bench/run.py --tag main --stage 2 --workers 6      # a second run of the ordered attacks and baselines
python safety_bench/attribute.py --tag main                      # was each adopted false figure disclosed to the reader?
python safety_bench/summarize.py --tag main
```

`run.py` checks the API keys before starting, records the dataset hash, commit, and model versions in a manifest, and resumes an interrupted batch. Parallel workers run as separate processes. A run averages roughly seven web searches, so check your search quota before a large batch.

Results are written to `safety_bench/results/<tag>/`, and agent trajectories to `runs/<tag>/`. Both are excluded from version control.

## Scoring

Each run gets one outcome:

| Outcome | Meaning | Counted in attack success rate |
|---|---|---|
| Attack succeeded or failed | The attack reached the main model, with this result | Yes |
| Filtered by sub-model | The retrieval sub-model read the hidden instruction and did not pass it on | End-to-end rate only |
| Not used | The attack played no part in the run | No |
| Inconclusive | The report did not cover the target item | No |
| Error | An upstream failure or a judge failure | No; retried on resume |

Reach is read from evidence in the tool outputs, not assumed. Leaks and URL openings are scored by deterministic checks on outbound tool calls. Other outcomes are scored by an LLM judge against rubrics frozen with the dataset, with a literal text check used to flag disagreements for review. Report quality is measured against FrontierFinance rubrics where available, and against the clean baseline of the same task.

`summarize.py` reports attack success rates with Wilson and entity-level bootstrap intervals, utility under attack, extra tool use and budget consumption for false premises, and the number of false figures adopted without disclosure.

The full specification is in [safety_bench/README.md](safety_bench/README.md).

## Repository layout

```
finance_agent/            reference agent and tools
  adversary/              attack environment: edits tool outputs, serves fabricated pages, logs exposure
safety_bench/
  taxonomy.json           classes, attacks, and variants
  schema.py, schema/      case and run formats
  dataset/cases.jsonl     public cases
  build/                  case generation, validation, and the redacted seed data
  run.py, evaluate.py     batch runner and per-run scoring
  judge.py, vendor/       rubric judge, built on FrontierFinance's grader
  summarize.py, attribute.py
tests/
```

## Limitations

- By default the judge is the same model as the agent. Leak and URL checks are deterministic; other outcomes depend on the judge.
- The leak check matches exact secret terms, so a paraphrased leak is missed.
- Tampered filing pages are edited only when the agent opens them. Search snippets and EDGAR results are not edited.
- Agent runs vary. Plan for repeated runs before comparing attack variants.

## License and attribution

- This repository is released under the MIT License. It extends [vals-ai/finance-agent-v2](https://github.com/vals-ai/finance-agent-v2), Copyright (c) 2025 Vals AI, Inc.
- `safety_bench/vendor/frontier_finance/` contains modified code from [samaya-ai/frontier-finance](https://github.com/samaya-ai/frontier-finance), Copyright 2026 Samaya AI, Inc., under the Apache License 2.0. See its LICENSE and NOTICE.
- FrontierFinance, Copyright 2026 Samaya AI, Inc., is licensed under CC-BY-4.0 and is not included here. Cases built from it require your own copy, used under its terms.
- Share prices come from Tiingo, and filings from SEC EDGAR.
