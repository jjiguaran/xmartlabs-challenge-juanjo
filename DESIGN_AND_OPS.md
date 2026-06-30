# Design & Operations

## Part 1: Evaluation Strategy

### Implemented: Recall@K

- **Ground truth (`ground_truth.py`)**: interactive tool — given a question + keyword, it keyword-searches all chunks in the vector store and shows each match for manual y/n inclusion. Saves question, keyword, accepted chunks, and source to `data/ground_truth.json`. Faster than hand-writing chunks, keeps a human judging actual relevance (not just keyword presence).
- **Metric (`evaluation.py`)**: `RAGEvaluator` retrieves top-K via the production `VectorStoreRAGPipeline` (K = `TOP_K` in `config.py`, currently 5), computes `Recall@K = |expected ∩ retrieved| / |expected|`, logs each run with timestamp to `data/eval_logs.json`.
- **Experimentation**: changing `TOP_K` or the embedding model in `config.py` and re-running logs the new parameters alongside the score in `eval_logs.json`, making retrieval config comparisons traceable over time.
- **Current state**: 1 ground-truth question logged. Tooling scales — adding more is just more `ground_truth.py` sessions.

**Known limitation**: relevance is exact string match, not chunk ID — brittle if chunking changes. Fix would be matching on chunk ID/offset instead of text.

### Other metrics (not implemented) — approach

**Faithfulness.** Score = `supported claims / total claims`, via LLM-as-judge: decompose answer into atomic claims, ask judge yes/no per claim against retrieved context, average.

Engineering pieces:
- **`JUDGE_MODEL`** in `config.py`, separate from the generation model — avoids self-grading. `temperature=0`, structured JSON output (claims + yes/no).
- **Test set JSON** (sibling to `ground_truth.json`): questions tied to specific manual concepts, a trap-question flag, and the judge prompt template/version used — `evaluation.py` takes it as input like it already does `ground_truth.json`.
- **`eval_logs.json`** extended with `question`, `faithfulness_score`, `generator_model`, `judge_model`, `judge_prompt_version` — any of those changing should re-baseline, so they need to be visible in history.

Dataset as **contrastive pairs**: normal questions + trap questions that sound in-scope but aren't covered. Traps should score low for free — no extra labeling, tests honesty not just correctness.

Runs in CI on prompt/model changes, ~10-20 cases per concept, with a threshold that fails the build on regression.

**Answer relevance.** Score = 1–5 judging whether the answer directly addresses the query's intent, is complete, and avoids filler — brevity matters here since these are student pilots in training.

Engineering pieces:
- Same `JUDGE_MODEL` + `temperature=0`. CoT prompt: deconstruct query intent → analyze completeness → output structured JSON with score, reasoning, and missing components.
- **Intent dataset JSON**: curated questions across three types — definitions, procedural sequences, troubleshooting. Same input pattern as faithfulness test set.
- **`eval_logs.json`** extended with `relevance_score`, `judge_reasoning`, `missing_components`, `token_cost` — token cost here because the CoT judge is more expensive than a yes/no call, worth tracking across runs.

Relevance drift (answers getting longer or less direct after a model/prompt update) is detectable from the log history without a separate process.

---

## Part 2: System Improvements

**Implemented: Role-Based Prompt Architecture**

Three agent roles defined in `src/prompts/roles.json`, each with a `title`, `initial_message`, and `prompt`. Roles are loaded at startup in `main.py` — user selects one via CLI (`make run-chat`) before the session begins. The selected role's prompt is injected as the system prompt for that session.

Current roles and their prompt strategy:
- **Aviation Expert** — chain-of-thought reasoning, admits uncertainty when context is insufficient
- **Safety Auditor** — strict grounding in the manual only, bulleted procedural format, no speculation
- **Flight Instructor** — analogies and step-by-step breakdowns, encouraging tone

Adding a new role requires only a new entry in `roles.json` — no code changes.

---

## Part 3: Production Roadmap

### Scalability

| Priority | Task | Component | Operational / ML Value |
|---|---|---|---|
| P0 | Containerize RAG pipeline | AWS Lambda / ECS | Auto-scaling inference, isolated dependencies, no idle cost |
| P0 | Migrate vector index | OpenSearch Serverless | Managed sharding, sub-100ms retrieval at any corpus size |
| P0 | CI/CD evaluation gate | GitHub Actions + `src/evaluation` | Runs eval pipeline on every commit, blocks deploy if Recall@K drops below threshold |
| P0 | Local input PII masking | Microsoft Presidio / LiteLLM Proxy | Scans and redacts names, student IDs, and emails locally before data leaves the VPC to third-party LLMs |
| P1 | Add traffic buffer | API Gateway + SQS | Decouples ingestion, absorbs concurrent spikes without cascade failure |
| P1 | Query caching layer | ElastiCache (Redis) | Bypasses LLM + vector DB on frequent queries, reduces latency and token cost |
| P1 | Jailbreak & prompt injection defense | Llama Guard 3 / LLM Guard | Fast classifier acting as a firewall — rejects non-aviation queries and malicious overrides, preserving token budget |
| P1 | Token-based user quotas | API Gateway + Redis | Hard daily token ceilings per student ID — prevents script abuse while ensuring fair access across the cohort |
| P1 | Telemetry & log aggregation | CloudWatch + OpenTelemetry | Captures query, retrieved chunk IDs, and response per request — feeds monitoring and drift detection |
| P2 | Continuous golden dataset evaluation | Cron Job + `src.evaluation` | Nightly Recall@K run against 100 core aviation questions in production — detects silent degradation from LLM provider updates |
| P2 | Shadow deployments / A/B testing | AWS AppConfig | Routes a traffic slice to a new model or prompt variant; compare production metrics before full rollout, enables fast rollback |
| P2 | Socratic mode toggle | System prompt architecture | Injects instructional guardrails (e.g. reference chapter, ask guiding questions rather than raw answers) — switchable per session |
| P2 | Static frontend | S3 + CloudFront | Global CDN, offloads compute bandwidth |