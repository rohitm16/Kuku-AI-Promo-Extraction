# Kuku AI Promo Discovery System
### Prototype Streamlit App and Promo Extraction Pipeline

A prototype that finds 20-to-60 second promo candidates in timestamped microdrama screenplays. It uses deterministic parsing and beat clustering, a heuristic shortlist, LLM scoring and diversity-aware ranking. See the solution document for known failures.

Built for the **Kuku AI Generalist Assignment — Problem 1: Find the Promos (Ads)**.

🔗 **GitHub Repository:** [https://github.com/rohitm16/Kuku-AI-Promo-Extraction](https://github.com/rohitm16/Kuku-AI-Promo-Extraction)

---

## 🌟 Executive Quickstart: Launching the Web Application

The system features an executive, interactive web frontend built with Streamlit:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit application
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Key Capabilities of the Web Application:
* **Interactive Discovery Dashboard:** Filter promos by episode, hook type (visual, audio_verbal, audiovisual_sync), score range and free-text search.
* **Two Rankings:** Diversity-adjusted (at most 1 promo per episode in the top 10 and 2 in the top 25) and unrestricted.
* **Script Slice Inspector:** Shows the candidate text and the next beat after the cut (median 5 seconds).
* **Model Configuration Hub:** Password-masked OpenAI key entry stored strictly in memory, verified model presets (`gpt-4o-mini`, `gpt-4o`, `o3-mini`, `gpt-4-turbo`), custom model ID entry, and a 1-click **Test API Connection** verification probe.
* **Pre-Run Resource & Cost Estimator:** Real-time token and financial cost estimation before dispatching OpenAI runs.
* **Safe Screenplay Ingestion:** Built-in 50-episode dataset loader, multi-file `.txt` upload, and ZIP archive extraction with strict **Zip-Slip path traversal prevention**.
* **12 Test Fixtures:** Hand-written cases that can be run on a live model. No valid result exists yet.
* **One-Click Deliverables Hub:** Download buttons for ranked JSON/CSV, raw evaluations (9MB), checkpoints, and telemetry.

---

## 1. Results From the Earlier 463-Candidate Run

All metrics reflect genuine execution on the full 50-episode screenplay dataset (*"I Married a Broken Soldier. He was a Duke"*):

| Metric | Verified Production Result | Notes & Verification Reference |
| :--- | :---: | :--- |
| **Episodes Ingested** | **50 / 50** | Episode 6 uses different header casing, so 2 events and its scene times were lost. |
| **Raw Multi-Track Events Extracted** | **3,725 events** | Dialogue, visual action, camera cues, and sound effects. |
| **Atomic Beats Formed** | **1,328 beats** | Clustered by speaker alternations and action boundaries. |
| **Valid 20–60s Windows Generated** | **8,730 windows** | Combinatorial slices strictly meeting $20.0\text{s} \le \Delta t \le 60.0\text{s}$. |
| **Shortlisted Candidates** | **463 candidates** | Selected across 3 pruning lanes (~$9.26$ candidates / episode). |
| **Genuine Remote LLM Evaluations** | **463 / 463 (100%)** | Evaluated via OpenRouter (`meta-llama/llama-3.3-70b-instruct`). |
| **Simulated / Mock Evaluations in Deliverables** | **0 (0.0%)** | Zero mock records in final deliverables (`is_simulated: false`). |
| **Duration Violations** | **0 (0.0%)** | Min: $20.00\text{s}$, Max: $60.00\text{s}$ (100.0% duration compliance). |
| **Intra-Episode Duplicates Suppressed** | **181 candidates** | Removed overlapping slices ($\text{IoU} > 0.50$) to prevent redundant ads. |
| **Final Ranked Promotional Clips** | **282 promos** | Exported in both Diversity-Adjusted and Unrestricted formats. |
| **Automated Tests** | **48** | 48 pass with the confidential dataset present. Without it, 11 data tests are skipped. |
| **Total Prompt Tokens Consumed** | **403,005 tokens** | Tracked via API telemetry across 155 batches of 3 candidates. |
| **Total Completion Tokens Consumed** | **100,584 tokens** | Structured Pydantic JSON evaluations. |
| **Inference Cost** | **$0.0785 at list price** | OpenRouter's usage counter read $0.193 before the last 25 candidates. |

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph UI [Streamlit Web Application Layer (ui/)]
        UI1["app.py: Dashboard & Multi-Tab Hub"]
        UI2["file_manager.py: Zip-Slip Safe Ingestion"]
        UI3["model_settings.py: Credential & Preset Manager"]
        UI4["run_controller.py: Real-Time Pipeline Runner"]
        UI5["results_view.py: Interactive Inspector & Filters"]
        UI6["audit_view.py: 12-Case Quality Inspector"]
        UI7["export_view.py: One-Click Artifact Downloads"]
    end

    subgraph S1 [Stage 1: Multi-Track Ingestion]
        A["50 Screenplay Text Files"] --> B["Deterministic Parser"]
        B --> C["Track Separation: Scene, Camera, Action, Dialogue, Sound"]
        C --> D["Timeline Synchronizer & Beat Clusterer"]
        D --> E["3,725 Events &rarr; 1,328 Atomic Beats"]
    end

    subgraph S2 [Stage 2: Combinatorial Windowing & Pruning]
        E --> F["Combinatorial Slicing (20.0s &le; &Delta;t &le; 60.0s)"]
        F --> G["8,730 Valid Windows"]
        G --> H1["Lane 1: Boundary Salience"]
        G --> H2["Lane 2: Development Acceleration"]
        G --> H3["Lane 3: Thematic Anomaly"]
        H1 & H2 & H3 --> I["Temporal Non-Maximum Suppression (IoU > 0.60)"]
        I --> J["463 Shortlisted Candidates (~9/ep)"]
    end

    subgraph S3 [Stage 3: LLM Evaluation]
        J --> K["Mini-Batch Packaging (Batch Size = 3)"]
        K --> L["Official OpenAI Client Adapter (kuku_promo_system/llm_client.py)"]
        L --> M["Cold-Viewer Prompt"]
        L --> N["Lookahead Resolution Audit (Post-Cut Probe)"]
        M & N --> O["Pydantic Structured Schema Validation"]
        O --> P["Model-Aware Checkpoint Persistence"]
    end

    subgraph S4 [Stage 4: Ranking & Deliverables]
        P --> Q["Composite Narrative Scoring & Penalties"]
        Q --> R["Intra-Episode Deduplication (IoU > 0.50: 463 &rarr; 282 Promos)"]
        R --> S1["Unrestricted Global Ranking"]
        R --> S2["Diversity-Adjusted Ranking (Max 1 in Top 10/Ep)"]
        S1 & S2 --> T["Deliverable Exports (JSON, CSV, Raw Evaluations, Audit)"]
    end

    UI4 --> S1
    S4 --> UI5
    S4 --> UI7
```

---

## 3. Mathematical Scoring Formulation

The final ranking metric is the **Composite Score** ($S_{\text{composite}}$), computed via a weighted base score modulated by binary penalty multipliers:

$$S_{\text{composite}} = \left(0.35 \cdot S_{\text{hook}} + 0.35 \cdot S_{\text{cliff}} + 0.30 \cdot S_{\text{clarity}}\right) \times P_{\text{lookahead}} \times P_{\text{lore}}$$

* **Hook Score ($S_{\text{hook}}$, 35%):** First 3-5 seconds scroll-stopping power on mobile feeds.
* **Cliffhanger Score ($S_{\text{cliff}}$, 35%):** Unresolved dramatic jeopardy at the cut point.
* **Standalone Clarity ($S_{\text{clarity}}$, 30%):** Self-sufficiency for uninitiated cold viewers.
* **Lookahead Penalty ($P_{\text{lookahead}} = 0.70$):** 30% penalty if the next beat after the cut answers or deflates the dramatic question.
* **Prior Lore Penalty ($P_{\text{lore}} = 0.50$):** 50% penalty applied if the conflict is incomprehensible without previous episode knowledge.

---

## 4. Repository Structure

```text
.
├── app.py                      # Executive Streamlit web application root
├── ui/                         # Modular Streamlit UI presentation layer
│   ├── __init__.py
│   ├── components.py           # Executive CSS, KPI cards, badges
│   ├── file_manager.py         # Ingestion, validation, Zip-Slip prevention
│   ├── model_settings.py       # Password masking, presets, connection testing
│   ├── run_controller.py       # Pipeline execution controller & cost estimator
│   ├── results_view.py         # Dashboard: filtering, table/card views, script viewer
│   ├── audit_view.py           # 12-case benchmark inspector & synthetic disclosure
│   └── export_view.py          # One-click artifact download hub
├── kuku_promo_system/          # Core production engine
│   ├── __init__.py
│   ├── config.py               # Constants, OpenAI presets, pricing metadata
│   ├── logger.py               # Structured logging with secret masking
│   ├── parser.py               # Multi-track screenplay regex parser
│   ├── timeline.py             # Event synchronization & atomic beat clustering
│   ├── pruner.py               # Combinatorial windowing & 3-lane pruner
│   ├── llm_client.py           # Official OpenAI adapter & circuit breakers
│   ├── evaluator.py            # Cold-Viewer evaluator & model-aware checkpointing
│   ├── ranker.py               # Deduplication, global ranking & CSV/JSON export
│   ├── audit.py                # 12-case synthetic benchmark quality suite
│   └── pipeline.py             # Unified command-line interface
├── tests/                      # Automated test suite (48 tests)
│   ├── test_foundation.py      # Duration assertions & secret masking
│   ├── test_parser.py          # Screenplay regex parsing
│   ├── test_timeline.py        # Event merging & beat splitting
│   ├── test_pruner.py          # Combinatorial windowing & 3-lane pruner
│   ├── test_evaluator.py       # Pydantic schema validation & scoring math
│   ├── test_ranker.py          # Temporal deduplication & diversity caps
│   ├── test_multi_provider_recovery.py # OpenAI resolution & circuit breakers
│   └── test_ui_and_safety.py   # Zip-Slip safety, cost estimator, checkpoint separation
├── docs/                       # Technical documentation & interview guides
│   ├── ARCHITECTURE.md         # Detailed module design & data flow
│   ├── SCORING_AND_EVALUATION.md # Mathematical rubric, penalties & schema
│   ├── REPRODUCIBILITY.md      # Setup, execution & verification guide
│   ├── STREAMLIT_APPLICATION.md# Streamlit UI guide & features
│   ├── OPENAI_CONFIGURATION.md # OpenAI setup, credentials & model presets
├── raw_outputs/                # Verified production outputs & checkpoints
│   ├── ranked_promos.json      # Diversity-adjusted ranked promos (JSON)
│   ├── ranked_promos.csv       # Diversity-adjusted ranked promos (CSV)
│   ├── ranked_promos_unrestricted.json # Unrestricted meritocratic ranking (JSON)
│   ├── ranked_promos_unrestricted.csv  # Unrestricted meritocratic ranking (CSV)
│   ├── raw_evaluations.json    # Complete 463-record evaluation provenance (9MB)
│   ├── eval_checkpoint.jsonl   # Canonical execution checkpoint
│   ├── benchmark_audit_report.json # 12-case quality audit report
│   ├── phase10_final_telemetry.json# Final token and cost telemetry
│   └── openai_run/             # Clean output destination for new OpenAI runs
├── submission/                 # Official recruitment submission deliverables
│   ├── Kuku_Find_the_Promos_Solution.pdf # Official 2-page solution brief
│   └── README.md               # Submission overview & deliverable index
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
├── .gitignore                  # Security guardrail excluding API keys
└── README.md                   # Main project landing page
```

---

## 5. Execution & Verification

### Run Complete Automated Test Suite (48 Tests)
```bash
python -m pytest tests/ -v
```
48 tests pass with the dataset present.

The code uses the OpenAI Python SDK as a client library. To use OpenRouter or another OpenAI-compatible endpoint, set OPENAI_BASE_URL and OPENAI_API_KEY in your terminal and pass --model, or enter the key and base URL in the Streamlit sidebar.

### Run Pipeline via CLI
* **Run 12-case Quality Audit:**
  ```bash
  python -m kuku_promo_system.pipeline --audit
  ```
* **Run Full 50 Episodes (CLI):**
  ```bash
  python -m kuku_promo_system.pipeline --all
  ```
* **Run Specific OpenAI Model:**
  ```bash
  python -m kuku_promo_system.pipeline --all --model gpt-4o
  ```

---

