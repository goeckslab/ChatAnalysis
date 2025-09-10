# DSPy Agent Evaluation — README

Two runners to evaluate **DA-Agent** style benchmarks:

- `eval_dspy.py` — uses your **DSPy agent**.
- `eval_biomni.py` — uses the **Biomni** agent.

Both enforce **tags-only** outputs (e.g., `@mean_fare[34.65]`) and write **timestamped** results.

---

## 1) Prerequisites

- Python **3.8+**
- DA-Agent data layout:
```
<DATA_ROOT>/
  da-dev-questions.jsonl
  da-dev-labels.jsonl
  data/datasets/csv/<all files referenced by file_name>
```

---

## 2) Quick Start

### dspy runner
```bash
python3 -m venv dspy_venv
source dspy_venv/bin/activate
pip install -r requirements_nicegui_dspy.txt

# minimal
python eval_dspy.py   --model openai/gpt-4o   --data-root ./benchmarks/DA-Agent

# with limits & pacing
python eval_dspy.py   --model openai/gpt-4o   --data-root ./benchmarks/DA-Agent   --limit 25 --sleep-seconds 2
```

### Biomni runner
```bash
python3 -m venv biomni_venv
source biomni_venv/bin/activate
pip install -r requirements_biomni.txt

# minimal
python eval_biomni.py   --llm "gpt-4o"   --data-root ./benchmarks/DA-Agent

# with details and pacing
python eval_biomni.py   --llm "claude-3-7-sonnet-20250219"   --data-root ./benchmarks/DA-Agent   --limit 25 --sleep-seconds 2   --save-details --details-on fail --details-limit 10   --print-details --print-on fail --print-limit 5
```

---

## 3) Environment & API Keys

Both runners load a `.env` automatically (via `python-dotenv`).

- Put them in `.env`.

---

## 4) Running `eval_dspy.py`

**Common flags**
- `--model` (e.g., `openai/gpt-4o`, `groq/llama3-70b-8192`, `google/gemini-2.5-pro`)
- `--data-root <DATA_ROOT>`
- `--limit N`, `--sleep-seconds S`
- `--compile-examples ./examples.json` (enable few-shot compilation)

**Re-run a subset**
- `--only-ids "474,480,490"` — run only these question IDs.
- `--only-ids-file <old_results.json>` — extract IDs from a prior run.
- `--reason-substr "RateLimitError"` — with `--only-ids-file`, filter by reason.

**Traces (how it solved it)**
- `--dump-on both|pass|fail|none` (default `both`)
- Caps: `--dump-limit 10` (total), `--dump-pass-cap 3`, `--dump-fail-cap 7`
- Periodic: `--dump-every N`
- Where: `--dump-dir evaluation_outputs/traces`

**Outputs**
- Results JSON path base: `--out evaluation_outputs/results.json`
- **Timestamp appended by default** → `results_<model>_<YYYYmmdd-HHMMSS>.json`
- Disable timestamp: `--no-timestamp-out`

**Files written**
- Results JSON: `evaluation_outputs/results_<model>_<YYYYmmdd-HHMMSS>.json`
- Traces JSONL: `evaluation_outputs/traces/traces_<model>_<YYYYmmdd-HHMMSS>.jsonl`

---

## 5) Running `eval_biomni.py`

**Common flags**
- `--llm "gpt-4o"` (or any Biomni-supported model)
- `--data-root <DATA_ROOT>`
- `--limit N`, `--offset K`, `--sleep-seconds S`
- Dataset layout overrides:  
  `--questions-fn`, `--labels-fn`, `--csv-subdir` (default: `data/datasets/csv`)

**Printing to console**
- `--print-details`  
- `--print-on fail|pass|all` (default `fail`)  
- `--print-limit 5`

**Saving a capped “details” JSON**
- `--save-details`  
- `--details-on fail|pass|all` (default `fail`)  
- `--details-limit 10`

**Files written**
- Main JSON: `evaluation_outputs/biomni_eval_<llm>_<YYYYmmdd_HHMMSS>.json`
- CSV: `evaluation_outputs/biomni_eval_<llm>_<YYYYmmdd_HHMMSS>.csv`
- Details JSON (optional): `evaluation_outputs/biomni_eval_<llm>_<YYYYmmdd_HHMMSS>_details.json`

---

## 6) Re-run only rate-limited failures (dspy)

From a prior results file:
```bash
python eval_dspy.py   --model openai/gpt-4o   --data-root ./benchmarks/DA-Agent   --only-ids-file evaluation_outputs/results_openai_gpt-4o_20250910-123000.json   --reason-substr "RateLimitError"   --sleep-seconds 2
```

Or direct IDs:
```bash
python eval_dspy.py   --model openai/gpt-4o   --data-root ./benchmarks/DA-Agent   --only-ids "474,480,490"   --sleep-seconds 2
```

---

## 7) Requirements

Keep **separate virtualenvs** for each runner.


## 8) Troubleshooting

- **Dataset file not found**  
  Ensure `<DATA_ROOT>/data/datasets/csv/<file_name>` exists for each question.

- **API errors / 401**  
  Confirm `.env` is loaded; keys exported; or pass `--api-key` (dspy).

- **Rate limits**  
  Use `--sleep-seconds`, lower `--limit`, and/or re-run with `--only-ids-file ... --reason-substr "RateLimitError"`.

- **FAIL due to tag mismatch**  
  The runners require **exact tag names** and **requested rounding**. Remove extra prose; output only `@tag[value]`.

- **Slow / timeouts**  
  Reduce `--limit`, increase sleep, or run a smaller subset of IDs.

---

## 9) Result Schema (overview)

Both runners produce a top-level JSON:
```json
{
  "summary": {
    "total": 123,
    "passed": 100,
    "accuracy": 0.813,
    "model": "openai/gpt-4o",
    "...": "runner-specific extras (timestamp, dump_path, etc.)"
  },
  "details": [
    {
      "id": 474,
      "status": "PASS|FAIL|ERROR|SKIP",
      "question": "...",
      "expected": "... or expected_tags map",
      "prediction|prediction_raw": "...",
      "dataset_path": "/abs/path/to.csv",
      "validation": {
        "expected_tags": {...},
        "predicted_tags": {...},
        "missing_keys": [],
        "mismatches": {"tag": {"expected": "...", "predicted": "..."}},
        "matched_keys": {...},
        "extra_keys": []
      },
      "latency_sec": 1.234,
      "feedback": "human-friendly diff",
      "reason": "only for ERROR/SKIP"
    }
  ]
}
```

---

**That’s it—happy benchmarking!**
