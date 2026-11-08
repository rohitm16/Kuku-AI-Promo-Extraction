# Architecture & System Design

This document details the engineering architecture, module responsibilities, data flow, candidate generation heuristics, evaluation schemas, and production safeguards of the **Kuku AI Promo Extraction System**.

---

## 1. System Architecture Overview

The system is engineered as a **four-stage deterministic-to-semantic funnel** coupled with a **modular Streamlit presentation layer**. High-volume chronological text is progressively filtered through computationally efficient rule-based stages before invoking large language models for deep tension evaluation.

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

    subgraph S1 [Stage 1: Multi-Track Screenplay Ingestion & Timeline]
        A["50 Screenplay Text Files<br/>(Raw Multi-Track Production Scripts)"] --> B["Multi-Track Deterministic Parser<br/>(kuku_promo_system/parser.py)"]
        B --> C["Track Separation: Scene, Camera, Action, Dialogue, Sound"]
        C --> D["Timeline Synchronizer & Atomic Beat Clusterer<br/>(kuku_promo_system/timeline.py)"]
        D --> E["3,725 Raw Events &rarr; 1,328 Atomic Beats"]
    end

    subgraph S2 [Stage 2: Combinatorial Windowing & 3-Lane Pruning]
        E --> F["Combinatorial Window Generator<br/>(20.0s &le; &Delta;t &le; 60.0s)"]
        F --> G["8,730 Valid Windows"]
        G --> H1["Lane 1: Boundary Salience<br/>(Scene terminals, hook events)"]
        G --> H2["Lane 2: Development Acceleration<br/>(Dialogue pacing, tempo shifts)"]
        G --> H3["Lane 3: Thematic Anomaly<br/>(Intimate tension, quiet reveals)"]
        H1 & H2 & H3 --> I["Temporal Non-Maximum Suppression<br/>(IoU > 0.60 intra-lane deduplication)"]
        I --> J["463 Shortlisted Candidates (~9/episode)"]
    end

    subgraph S3 [Stage 3: Cold-Viewer LLM Evaluation]
        J --> K["Mini-Batch Packaging<br/>(Batch size = 3 candidates)"]
        K --> L["Official OpenAI Client Adapter<br/>(kuku_promo_system/llm_client.py)"]
        L --> M["Cold-Viewer Prompt<br/>(Clip text, next beat, batch-mates)"]
        L --> N["Lookahead Resolution Audit<br/>(Next-Beat post-cut context check)"]
        M & N --> O["Pydantic Structured Validation<br/>(SinglePromoEvaluation schema)"]
        O --> P["Model-Aware Checkpoint Persistence<br/>(raw_outputs/openai_run/eval_checkpoint.jsonl)"]
    end

    subgraph S4 [Stage 4: Ranking, Diversification & Deliverables]
        P --> Q["Composite Narrative Scoring<br/>(Base score &times; Lookahead &times; Lore penalties)"]
        Q --> R["Intra-Episode Deduplication<br/>(IoU > 0.50 suppression: 463 &rarr; 282 promos)"]
        R --> S1["Unrestricted Global Ranking<br/>(Strict score order)"]
        R --> S2["Diversity-Adjusted Ranking<br/>(Max 1 in Top 10, Max 2 in Top 25 per episode)"]
        S1 & S2 --> T["Deliverable Exports<br/>(JSON, CSV, Raw Provenance, Audit)"]
    end

    UI4 --> S1
    S4 --> UI5
    S4 --> UI7
```

---

## 2. Module Responsibilities

The codebase is organized into clean, modular Python packages under [`kuku_promo_system/`](file:///d:/Companies/Kuku/kuku_promo_system) and [`ui/`](file:///d:/Companies/Kuku/ui):

### Core Engine (`kuku_promo_system/`)

| Module | Source File | Core Responsibility |
| :--- | :--- | :--- |
| **Configuration** | [`config.py`](file:///d:/Companies/Kuku/kuku_promo_system/config.py) | Defines global constants, hard duration bounds ($20.0\text{s} \le \Delta t \le 60.0\text{s}$), OpenAI presets (`gpt-4o-mini`, `gpt-4o`, `o3-mini`, `gpt-4-turbo`), token limits, pricing metadata, and secret masking. |
| **Logger** | [`logger.py`](file:///d:/Companies/Kuku/kuku_promo_system/logger.py) | Provides structured console and file logging with automatic API key sanitization (`mask_secret()`). |
| **Parser** | [`parser.py`](file:///d:/Companies/Kuku/kuku_promo_system/parser.py) | Deterministic regex parser extracting time ranges, scene headers, speaker turns, action lines, and sound effects from screenplay files. |
| **Timeline** | [`timeline.py`](file:///d:/Companies/Kuku/kuku_promo_system/timeline.py) | Merges concurrent multi-track events, groups sequential lines into atomic dramatic beats, and isolates scene boundaries. |
| **Pruner** | [`pruner.py`](file:///d:/Companies/Kuku/kuku_promo_system/pruner.py) | Generates all valid 20–60s candidate windows, partitions them into 3 heuristic lanes, and performs temporal Non-Maximum Suppression (NMS). |
| **LLM Client** | [`llm_client.py`](file:///d:/Companies/Kuku/kuku_promo_system/llm_client.py) | Official OpenAI SDK adapter (`openai.OpenAI`), supporting verified presets, custom model IDs, connection verification probes, authentication circuit breakers, and bounded exponential backoff retries. |
| **Evaluator** | [`evaluator.py`](file:///d:/Companies/Kuku/kuku_promo_system/evaluator.py) | Formats batch evaluation prompts, executes Cold-Viewer and lookahead audits, validates Pydantic schemas, and manages model-aware checkpointing. |
| **Ranker** | [`ranker.py`](file:///d:/Companies/Kuku/kuku_promo_system/ranker.py) | Applies intra-episode temporal deduplication ($\text{IoU} \le 0.50$), builds unrestricted and diversity-adjusted rankings, and exports JSON/CSV. |
| **Audit** | [`audit.py`](file:///d:/Companies/Kuku/kuku_promo_system/audit.py) | Implements the 12-case synthetic benchmark quality suite to detect duration non-compliance, lore traps, and artificial cliffhangers. |
| **Pipeline CLI** | [`pipeline.py`](file:///d:/Companies/Kuku/kuku_promo_system/pipeline.py) | Unified command-line interface supporting full dataset runs (`--all`), single episodes (`--episode`), audit mode (`--audit`). |

### Modular Presentation Layer (`ui/`)

| Module | Source File | Core Responsibility |
| :--- | :--- | :--- |
| **Application Hub** | [`app.py`](file:///d:/Companies/Kuku/app.py) | Executive Streamlit root application integrating all six operational tabs and sidebar configuration. |
| **Components** | [`ui/components.py`](file:///d:/Companies/Kuku/ui/components.py) | Custom executive CSS typography, KPI metric cards, and live/mock status badges. |
| **File Manager** | [`ui/file_manager.py`](file:///d:/Companies/Kuku/ui/file_manager.py) | Ingestion service supporting built-in dataset discovery, multi-file TXT validation, and Zip-Slip safe extraction. |
| **Model Settings** | [`ui/model_settings.py`](file:///d:/Companies/Kuku/ui/model_settings.py) | Sidebar controls for masked OpenAI key input, preset dropdowns, custom model ID entry, and connection testing. |
| **Run Controller** | [`ui/run_controller.py`](file:///d:/Companies/Kuku/ui/run_controller.py) | Pipeline execution manager with pre-run cost estimation, real-time stage progress bars, and batch activity feeds. |
| **Results View** | [`ui/results_view.py`](file:///d:/Companies/Kuku/ui/results_view.py) | Interactive promo dashboard with multi-criteria filtering (episode, hook type, score), keyword search, and script inspector. |
| **Audit View** | [`ui/audit_view.py`](file:///d:/Companies/Kuku/ui/audit_view.py) | Synthetic 12-case benchmark inspector with disclosure banners and live quality re-testing capability. |
| **Export View** | [`ui/export_view.py`](file:///d:/Companies/Kuku/ui/export_view.py) | One-click deliverable download hub for JSON, CSV, JSONL checkpoints, and telemetry reports. |

---

## 3. Data Flow & Stage Transitions

### Stage 1: Screenplay Ingestion to Atomic Beats
1. **Raw Screenplay Text:** Multi-track files with timestamps formatted as `Start: (MM:SS.CC) | End: (MM:SS.CC)`.
2. **Event Extraction:** Each block is categorized as `scene_header`, `dialogue`, `action_visual`, `camera_direction`, or `sound_fx`.
3. **Atomic Beat Synthesis:** Consecutive dialogue turns between speakers or concurrent visual actions within a temporal gap ($\le 0.8\text{s}$) are clustered into an `AtomicBeat`. Across all 50 episodes, 3,725 raw events condense into 1,328 atomic beats.

### Stage 2: Combinatorial Windowing & 3-Lane Pruning
1. **Valid Window Generation:** An exhaustive scan across all sequential combinations of atomic beats $[B_i, B_j]$ within an episode is performed. Only windows satisfying $20.0\text{s} \le \Delta t \le 60.0\text{s}$ are retained (yielding 8,730 valid windows across 50 episodes).
2. **Three-Lane Heuristic Partitioning:**
   - **Boundary Salience Lane:** Favors windows terminating on scene endings or sudden action cutoffs (target: 4 candidates/ep).
   - **Development Lane:** Favors windows with rapid dialogue alternations and high pacing acceleration (target: 4 candidates/ep).
   - **Anomaly Lane:** Counts high-stakes keywords and close-up shots (target: 3 candidates/ep).
3. **Temporal Non-Maximum Suppression (NMS):** Overlapping candidates within the same lane with $\text{IoU} > 0.60$ are suppressed, shortlisting exactly **463 candidates** (~9.26 candidates per episode).

### Stage 3: Cold-Viewer LLM Evaluation
1. **Batch Packaging:** Candidates are packaged in batches of 3 to maximize token efficiency and throughput.
2. **Context Isolation:** The prompt instructs the model to use only the clip. The next beat and two other candidates appear in the same prompt.
3. **Next-Beat Lookahead Probe:** Each candidate prompt includes the immediate post-cut transcript slice $[T_{end}, T_{next\_beat}]$. The model determines whether the dramatic question is immediately deflated.
4. **Structured Schema Validation:** Responses are validated via Pydantic (`SinglePromoEvaluation`).
5. **Persistent Checkpointing:** Each record is appended to `eval_checkpoint.jsonl` immediately upon return, guaranteeing resumption without re-evaluation.

### Stage 4: Global Ranking & Diversity Guardrails
1. **Composite Narrative Scoring:** Base score combines Hook (35%), Cliffhanger (35%), and Standalone Clarity (30%), multiplied by lookahead ($0.70$) and lore ($0.50$) penalty factors.
2. **Intra-Episode Deduplication ($\text{IoU} \le 0.50$):** Suppresses overlapping slices within an episode, reducing 463 evaluated candidates to **282 unique ranked promotional clips**.
3. **Dual Ranking Regimes:**
   - **Unrestricted:** Strict global descending sort by composite score.
   - **Diversity-Adjusted:** Enforces a maximum of 1 clip in the Top 10 and a maximum of 2 clips in the Top 25 per episode, ensuring ad creative variety across the catalog.

---

## 4. Production Engineering Safeguards

1. **Strict Duration Invariant ($20.0\text{s} \le \Delta t \le 60.0\text{s}$):**
   - Validated at combinatorial windowing.
   - No violations in the stored 463 candidates.

2. **Zip-Slip Path Traversal Protection:**
   - The file upload manager checks every archived ZIP entry for directory traversal sequences (`..`), leading path separators, and ensures target destination paths remain strictly within the sandboxed extraction root.

3. **Model-Aware Checkpoint Separation:**
   - Evaluator checkpoint loader checks both `evaluation_source` and `model_id`. A run initialized on `gpt-4o` will never load cached records evaluated on `gpt-4o-mini` or `meta-llama/llama-3.3-70b-instruct`, eliminating cross-model cache pollution.

4. **Historical Provenance Isolation:**
   - All 463 historical OpenRouter evaluation records remain preserved in `raw_outputs/`.
   - New live runs executed via OpenAI are routed to `raw_outputs/openai_run/`.

5. **Financial Circuit Breakers:**
   - Authentication errors (HTTP 401) and insufficient quota errors (HTTP 402/insufficient_quota) trigger immediate pipeline halt without blind retries, protecting user funds.
