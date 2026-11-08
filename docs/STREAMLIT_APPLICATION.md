# Streamlit Web Application Documentation

This document provides a comprehensive guide to the **Kuku AI Promo Discovery Web Application**, built with Streamlit and organized into modular presentation components under [`ui/`](file:///d:/Companies/Kuku/ui).

---

## 1. Overview & Launching

The web application provides an executive, user-friendly interface for content editors, marketing teams, and engineering reviewers to explore extracted promotional clips, launch pipeline runs, inspect candidate scripts, review quality benchmarks, and download deliverables.

### Quick Launch Command
```bash
streamlit run app.py
```
Default URL: `http://localhost:8501`

---

## 2. Architecture & File Structure

The application separates presentation logic into modular files under `ui/`:

```text
├── app.py                   # Main entry point and layout orchestrator
└── ui/
    ├── __init__.py          # Package initialization
    ├── components.py        # Custom CSS styling, header badges, KPI metric cards
    ├── file_manager.py      # Multi-track file ingestion & Zip-Slip safe extraction
    ├── model_settings.py    # Sidebar controls: key masking, presets, connection probe
    ├── run_controller.py    # Pipeline runner with progress tracking & cost estimator
    ├── results_view.py      # Results dashboard: table & card views, filters, script preview
    ├── audit_view.py        # 12-case benchmark inspector with synthetic disclosure
    └── export_view.py       # One-click artifact download manager
```

---

## 3. Sidebar Controls & Model Management

Located on the left sidebar:

1. **OpenAI API Key Input:**
   - Password-masked input field (`type="password"`).
   - Stored strictly in memory within `st.session_state["openai_api_key"]`.
   - Never logged to console, never written to disk, and never exported in deliverables.

3. **Model Presets & Custom Model Input:**
   - Dropdown with verified presets:
     - `gpt-4o-mini` (Recommended: fast, high quality, \$0.15 / \$0.60 per 1M tokens)
     - `gpt-4o` (Flagship multimodal model)
     - `o3-mini` (Fast reasoning model)
     - `gpt-4-turbo` (High-capacity model)
     - `Custom Model ID...` (Allows entering any custom model string)
   - Real-time token pricing display.

4. **Max Completion Tokens Slider:**
   - Adjustable between 500 and 3,000 tokens (default: 1,500 tokens).

5. **"Test API Connection" Verification Button:**
   - Sends a lightweight 1-token probe to OpenAI's `/chat/completions` endpoint.
   - Confirms valid credentials, positive quota balance, and model availability within seconds.

---

## 4. Main Navigation Tabs

The interface is structured into six intuitive tabs:

### Tab 1: 🏛️ System Overview & Architecture
* High-level summary of the Kuku AI Promo Discovery System.
* Top-level KPI cards summarizing canonical corpus (50 episodes), shortlisted candidates (463), duration compliance (100%), and total production spend (\$0.07854 USD).
* Mermaid flowchart illustrating the 4-phase deterministic-to-semantic funnel.
* Detailed explanation of multi-track parsing, atomic beat clustering, 3-lane pruning, cold-viewer LLM evaluation, and diversity-adjusted ranking.

### Tab 2: 🚀 Screenplay Ingestion & Pipeline Runner
* **Three Ingestion Sources:**
  - *Built-in 50-Episode Dataset:* Ingests canonical screenplays directly from local project files.
  - *Upload Custom Screenplay Files (.txt):* Accepts multi-track text files with real-time format validation.
  - *Upload ZIP Archive (.zip):* Safely extracts files with strict Zip-Slip path traversal prevention.
* **Pre-Run Resource & Cost Estimator:**
  - Automatically calculates candidate count, batch count, prompt tokens, completion tokens, and estimated USD cost based on selected model pricing before dispatching requests.
* **Live Pipeline Execution:**
  - Step-by-step visual progress bar (0% to 100%).
  - Real-time batch progress indicator showing candidate counts, success counts, and failures.

### Tab 3: 📊 Promotional Discovery Dashboard
* **Dataset Selector:** Toggle between the historical 463-candidate benchmark and recent OpenAI pipeline runs.
* **Ranking View:** Switch between *Diversity-Adjusted* (at most 1 promo per episode in the top 10 and 2 in the top 25) and *Unrestricted Global*.
* **Multi-Criteria Filter Bar:**
  - Filter by Episode number (All or specific episode).
  - Filter by Hook Archetype (`visual`, `audio_verbal`, `audiovisual_sync`, etc.).
  - Slider for composite score range (e.g. 7.5 to 10.0).
  - Free-text search matching keywords in rationale, conflict description, or hook description.
* **Display Modes:**
  - *Executive Card View:* Displays hook score & description, cliffhanger question & resolution flag, standalone conflict, editorial rationale, and expandable screenplay script slice with lookahead context.
  - *Compact Table View:* Sortable, formatted table for rapid review.

### Tab 4: 📜 Historical Benchmark (463 Candidates)
* Transparent record of the 50-episode evaluation on OpenRouter / Llama 3.3 70B.
* Clear disclosure that all 463 records are 100% genuine live evaluations generated at an investment of \$0.07854 USD.
* Breakdown of episode candidate yields, pruning lane distributions, and diversity caps.

### Tab 5: 🛡️ Quality Audit & 12-Case Benchmark
* Banner stating the earlier 100% result came from a simulator and is invalid.
* No results are shown until the fixtures are run on a live model.
* Case-by-case diagnostic table for all 12 benchmark test cases.
* Expandable category deep-dives explaining why each test case exists.
* **"🧪 Re-run Quality Audit" Button:** Allows running the 12-case benchmark live on demand.

### Tab 6: 📥 Exports & Deliverables
* One-click download buttons for all project artifacts:
  - `ranked_promos.json` & `ranked_promos.csv`
  - `ranked_promos_unrestricted.json` & `ranked_promos_unrestricted.csv`
  - `raw_evaluations.json` (Full 9MB candidate records)
  - `eval_checkpoint.jsonl` (Append-only checkpoint stream)
  - `benchmark_audit_report.json`
  - `phase10_final_telemetry.json`
* File size indicators and deliverable descriptions.

---

## 5. Security & Ingestion Safeguards

* **Zip-Slip Mitigation:** The file manager prevents directory traversal attacks by validating resolved member paths against the destination directory root.
* **Secret Protection:** API keys remain strictly in memory during the browser session and are never written to disk or included in exported deliverables.
* **Provenance Isolation:** Active OpenAI runs are routed cleanly to `raw_outputs/openai_run/` to ensure historical baseline records in `raw_outputs/` are never overwritten.
