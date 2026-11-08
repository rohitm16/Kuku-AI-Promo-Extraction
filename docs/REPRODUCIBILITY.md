# Reproducibility & Execution Guide

This guide provides step-by-step instructions for running, testing, and verifying the **Kuku AI Promo Extraction System** across different platforms (Linux, macOS, Windows), including both the **Interactive Streamlit Web Application** and the **Production CLI Pipeline**.

---

## 1. System Requirements

* **Python Version:** Python 3.10, 3.11, 3.12, 3.13, or 3.14 (Tested and verified on Python 3.14.7).
* **Operating System:** Platform-agnostic (Windows PowerShell, Linux Bash, macOS zsh).
* **Network Access:** Required only if running live remote LLM inference via OpenAI. (Offline testing and mock evaluation require zero network connectivity).

---

## 2. Installation & Setup

### Step 1: Clone or Extract Repository
```bash
cd kuku_promo_system_repo
```

### Step 2: Create and Activate Virtual Environment
**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Configuration & OpenAI Setup

Copy the example environment configuration template:
```bash
cp .env.example .env
```

### Option A: Running with Official OpenAI API (Live Mode)
Edit `.env` (or enter key directly into the Streamlit UI):
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-...your_openai_key_here...
OPENAI_MODEL=gpt-4o-mini
MAX_COMPLETION_TOKENS=1500
```
Verified models supported:
* `gpt-4o-mini` (Default recommended: high intelligence, ultra-low cost)
* `gpt-4o` (Flagship multimodal model)
* `o3-mini` (Fast reasoning model)
* `gpt-4-turbo` (High-capacity model)
* *Any custom OpenAI model slug via UI or CLI*

---

## 4. Launching the Interactive Streamlit Web Application

The system features an executive, interactive web frontend built with Streamlit:

```bash
streamlit run app.py
```

Once launched, open `http://localhost:8501` in your browser. The application provides:
1. **System Overview:** Architecture flowcharts, core metrics, and live configuration badges.
2. **Screenplay Ingestion & Runner:** Ingest built-in 50 episodes, upload custom `.txt` files, or upload `.zip` archives with Zip-Slip path traversal protection. Features pre-run cost estimation.
3. **Discovery Dashboard:** Filter promotional clips by episode, hook type, and composite score range; search text; inspect candidate script slices and post-cut lookahead context.
4. **Historical Benchmark:** Permanent byte-for-byte record of the 463-candidate OpenRouter production run.
5. **Quality Audit:** 12-case benchmark inspection with live re-test button and synthetic disclosure banner.
6. **Exports & Deliverables:** One-click download buttons for ranked JSON/CSV, raw evaluations, checkpoints, and telemetry.

---

## 5. Automated Test Suite (48 Tests)

Run the complete test suite verifying parsing, timeline synchronization, duration bounds, lane pruning, scoring math, circuit breakers, Zip-Slip safety, cost estimation, reasoning model compatibility (`max_completion_tokens`), and model-aware checkpointing:

```bash
python -m pytest tests/ -v
```

Expected output:
```text
============================= 48 passed ==============================
```

---

## 6. Command-Line Execution (CLI)

The modular pipeline can also be run directly from the terminal:

### 1. Run Complete Automated Benchmark Audit
Executes automated diagnostic checks across ground-truth hits, mundane scenes, lore traps, and artificial cliffhangers:
```bash
python -m kuku_promo_system.pipeline --audit
```

### 2. Run Pipeline on All 50 Screenplay Episodes
```bash
python -m kuku_promo_system.pipeline --all
```

### 3. Run Pipeline on a Specific Model
```bash
python -m kuku_promo_system.pipeline --all --model gpt-4o
```

### 4. Run Pipeline on a Single Episode (e.g., Episode 1)
```bash
python -m kuku_promo_system.pipeline --episode 1
```

### 5. Run Pipeline on an Episode Range (e.g., Episodes 1 to 5)
```bash
python -m kuku_promo_system.pipeline --start-ep 1 --end-ep 5
```

---

## 7. Inspecting Deliverables & Output Locations

* **Historical 463 Benchmark Outputs:** `raw_outputs/`
  - `ranked_promos.json` & `ranked_promos.csv` (Diversity-adjusted top promos)
  - `ranked_promos_unrestricted.json` & `ranked_promos_unrestricted.csv` (Global greedy rankings)
  - `raw_evaluations.json` (Full 9MB candidate evaluation records)
  - `eval_checkpoint.jsonl` (Append-only checkpoint stream)
  - `benchmark_audit_report.json` (12-case benchmark report)
  - `phase10_final_telemetry.json` (Telemetry & timing summary)

* **Active OpenAI Pipeline Runs:** `raw_outputs/openai_run/`
  - Checkpoints and deliverables are cleanly routed to this isolated folder to guarantee that historical baseline provenance is never overwritten.
