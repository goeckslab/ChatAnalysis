"""
Evaluate the DA-Agent benchmark using the Biomni agent.

Requirements:
  pip install biomni python-dotenv pandas tqdm

Notes:
  - Biomni can execute Python code and access local files. We leverage this by
    prompting it to read the dataset from an ABSOLUTE path and to output ONLY
    the benchmark-required tags (e.g., @mean_fare[34.65]) with no extra text.
  - This script does NOT depend on DSPy. It talks directly to Biomni.

Example:
  python eval_biomni.py \
    --llm "claude-3-7-sonnet-20250219" \
    --data-root ./benchmarks/DA-Agent \
    --limit 10 \
    --sleep-seconds 60 \
    --save-details \
    --details-on fail \
    --details-limit 10 \
    --print-details --print-on fail --print-limit 5

Environment:
  Biomni supports multiple providers. Set the matching API key(s), e.g.:
    export ANTHROPIC_API_KEY="..."
    export OPENAI_API_KEY="..."
    export GEMINI_API_KEY="..."
  Optionally place them in a .env file.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

from dotenv import load_dotenv
from tqdm import tqdm

# Biomni
try:
    from biomni.agent import A1
    from biomni.config import default_config
except Exception as e:
    print("ERROR: Failed to import Biomni. Install it first: pip install biomni", file=sys.stderr)
    raise

# --------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger("eval_biomni")

# To ensure we still write results on Ctrl-C
_SHOULD_FLUSH_AND_EXIT = False


def _sigint_handler(signum, frame):
    global _SHOULD_FLUSH_AND_EXIT
    _SHOULD_FLUSH_AND_EXIT = True
    logger.warning("SIGINT received: finishing current item and flushing outputs...")


signal.signal(signal.SIGINT, _sigint_handler)

# --------- Helpers ----------
TAG_RE = re.compile(r"@([A-Za-z0-9_]+)\[(.*?)\]")

def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Skipping malformed JSONL line in {path.name}: {e}")
    return items


def extract_expected_map(labels_map: Dict[int, dict], qid: int) -> Dict[str, str]:
    """
    labels_map[id] has {"id": int, "common_answers": [[key, value], ...]}
    Return a dict key->value (as strings).
    """
    expected = {}
    l = labels_map.get(qid)
    if not l:
        return expected
    for kv in l.get("common_answers", []):
        if not isinstance(kv, (list, tuple)) or len(kv) != 2:
            continue
        k, v = kv
        expected[str(k)] = str(v)
    return expected


def parse_predicted_tags(text: str) -> Dict[str, str]:
    """Parse @tag[value] pairs from the model output."""
    results: Dict[str, str] = {}
    for m in TAG_RE.finditer(text):
        key = m.group(1).strip()
        val = m.group(2).strip()
        results[key] = val
    return results


def looks_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _has_fixed_decimals(s: str) -> bool:
    # Detect fixed-decimal formatting like "1.2345" (not "1e-3")
    return "." in s and "e" not in s.lower()


def numeric_close(pred: str, exp: str) -> bool:
    """
    Compare numbers with a tolerance:
      - If expected uses fixed decimals: absolute tolerance = 0.5 * 10^-decimals
      - Else (integers or scientific notation): relative tolerance = 1e-6 (with small absolute floor)
    """
    try:
        p = float(pred)
        e = float(exp)
    except Exception:
        return False

    if _has_fixed_decimals(exp):
        decs = len(exp.split(".")[1])
        tol = 0.5 * (10 ** (-max(decs, 0)))
        return abs(p - e) <= tol + 1e-12

    # integers or scientific notation → relative tolerance
    rel = 1e-6
    abs_floor = 1e-9
    return abs(p - e) <= max(rel * max(1.0, abs(e)), abs_floor)


@dataclass
class EvalItem:
    id: int
    question: str
    constraints: str
    fmt: str
    file_name: str
    dataset_path: Path
    expected_map: Dict[str, str]


def build_prompt(item: EvalItem) -> str:
    """
    Build a strict prompt that instructs Biomni to:
    - Execute Python to read CSV at ABSOLUTE path
    - Follow constraints
    - Output ONLY tags required by 'fmt' string (no explanations)
    """
    rules = f"""
You are a strict data-analysis agent with Python code execution.
Load the dataset from the EXACT absolute path:
{str(item.dataset_path)}

TASK:
{item.question}

CONSTRAINTS (obey exactly):
{item.constraints}

OUTPUT FORMAT (obey EXACTLY, tags and brackets):
{item.fmt}

STRICT RULES:
- You MUST execute Python code to compute the result from the dataset at the exact path above.
- Use pandas/numpy/scipy/sklearn as needed.
- Round exactly as requested in the constraints/format.
- If "population standard deviation" is requested, use ddof=0.
- If Shapiro-Wilk is requested, use scipy.stats.shapiro with alpha=0.05.
- If Pearson correlation is requested, compute r and p-value using scipy or pandas.
- If an encoding or split is requested, follow it exactly.
- DO NOT guess numbers. Compute them from the CSV.
- OUTPUT ONLY the final tags exactly like "@tag[value]". No extra text, no explanations, no JSON.
- If multiple tags are required, output them separated by a single space in one line (preferred) or by newlines. No additional words.
- Do not include units or any extra punctuation beyond what the format shows.

Now compute and PRINT ONLY the final tags.
"""
    return rules.strip()


def coerce_output_to_text(resp: Any) -> str:
    """Biomni's A1.go(...) may return various structures; try best-effort text extraction."""
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    # Try dict-like common keys first
    if isinstance(resp, dict):
        for k in ["final", "answer", "text", "output", "content", "result", "message", "stream"]:
            v = resp.get(k)
            if isinstance(v, str):
                return v
        try:
            return json.dumps(resp, ensure_ascii=False)
        except Exception:
            return str(resp)
    # Try attribute access (SDK objects)
    for k in ["final", "answer", "text", "output", "content", "result", "message"]:
        try:
            v = getattr(resp, k, None)
            if isinstance(v, str):
                return v
        except Exception:
            pass
    return str(resp)


def evaluate_prediction(
    pred_map: Dict[str, str],
    expected_map: Dict[str, str]
) -> Tuple[bool, Dict[str, str], Dict[str, Tuple[str, str]], List[str], List[str]]:
    """
    Return (is_pass, matched_keys, mismatches, missing_keys, extra_keys)
    Pass = all expected keys exist AND match.
    """
    matched: Dict[str, str] = {}
    mismatches: Dict[str, Tuple[str, str]] = {}
    missing: List[str] = []

    for k, exp in expected_map.items():
        if k not in pred_map:
            missing.append(k)
            continue
        pred = pred_map[k]
        if looks_float(exp) and looks_float(pred):
            if numeric_close(pred, exp):
                matched[k] = pred
            else:
                mismatches[k] = (pred, exp)
        else:
            if str(pred).strip().lower() == str(exp).strip().lower():
                matched[k] = pred
            else:
                mismatches[k] = (pred, exp)

    extra = sorted(set(pred_map.keys()) - set(expected_map.keys()))
    is_pass = (len(missing) == 0 and len(mismatches) == 0)
    return is_pass, matched, mismatches, missing, extra


def feedback_text(
    pred_map: Dict[str, str],
    matched: Dict[str, str],
    mismatches: Dict[str, Tuple[str, str]],
    missing: List[str],
    extra: List[str],
) -> str:
    parts = []
    if missing:
        parts.append(f"Missing keys: {', '.join(missing)}")
    if mismatches:
        parts.append(
            "Mismatches: " + "; ".join([f"{k} (pred={p}, exp={e})" for k, (p, e) in mismatches.items()])
        )
    if extra:
        parts.append(f"Unexpected keys predicted: {', '.join(extra)}")
    if not parts:
        parts.append("All required keys matched.")
    return " | ".join(parts)


@dataclass
class EvalRunConfig:
    timestamp: str
    print_details: bool
    print_on: str
    print_limit: int
    save_details: bool
    details_on: str
    details_limit: int


def load_benchmark(
    data_root: Path,
    questions_fn: str = "da-dev-questions.jsonl",
    labels_fn: str = "da-dev-labels.jsonl",
    csv_subdir: str = os.path.join("data", "datasets", "csv"),
):
    questions_path = data_root / questions_fn
    labels_path = data_root / labels_fn

    if not questions_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Benchmark files not found under {data_root}. "
            f"Expected {questions_fn} and {labels_fn}."
        )

    questions = read_jsonl(questions_path)
    labels_raw = read_jsonl(labels_path)
    labels_map = {int(x["id"]): x for x in labels_raw if "id" in x}

    base_csv = (data_root / csv_subdir)

    items: List[EvalItem] = []
    for q in questions:
        # defensively handle missing keys
        try:
            qid = int(q["id"])
            file_name = q["file_name"]
        except Exception:
            logger.warning(f"Skipping malformed question row: {q}")
            continue

        ds_path = (base_csv / file_name).resolve()
        exp_map = extract_expected_map(labels_map, qid)
        items.append(
            EvalItem(
                id=qid,
                question=q.get("question", ""),
                constraints=q.get("constraints", ""),
                fmt=q.get("format", ""),
                file_name=file_name,
                dataset_path=ds_path,
                expected_map=exp_map,
            )
        )
    return items, labels_map


def save_main_reports(results: List[dict], out_dir: Path, model: str, timestamp: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    summary = {
        "summary": {
            "total": total,
            "passed": passed,
            "accuracy": (passed / total) if total else 0.0,
            "model": model,
            "timestamp": timestamp,
        },
        "details": results,
    }
    json_path = out_dir / f"biomni_eval_{model.replace('/', '_')}_{timestamp}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # CSV for quick spreadsheet eyeballing
    csv_path = out_dir / f"biomni_eval_{model.replace('/', '_')}_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id",
            "status",
            "question",
            "expected_keys",
            "prediction_raw",
            "dataset_path",
            "missing_keys",
            "mismatch_keys",
            "matched_keys",
            "extra_keys",
            "latency_sec",
            "feedback",
        ])
        for r in results:
            writer.writerow([
                r["id"],
                r["status"],
                r["question"],
                ";".join([f"{k}={v}" for k, v in r["expected"].items()]),
                r["prediction_raw"],
                r["dataset_path"],
                ";".join(r["validation"]["missing_keys"]),
                ";".join([f"{k}=>{p}|{e}" for k, (p, e) in r["validation"]["mismatches"].items()]),
                ";".join([f"{k}={v}" for k, v in r["validation"]["matched_keys"].items()]),
                ";".join(r["validation"]["extra_keys"]),
                f'{r.get("latency_sec", 0):.3f}',
                r.get("feedback", ""),
            ])

    logger.info(f"Saved JSON: {json_path}")
    logger.info(f"Saved CSV : {csv_path}")


def save_details_json(details: List[dict], out_dir: Path, model: str, timestamp: str, on: str, limit: int):
    if not details:
        logger.info("No detail samples collected; skipping details JSON.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "timestamp": timestamp,
        "criteria": {"on": on, "limit": limit},
        "count": len(details),
        "samples": details,
    }
    path = out_dir / f"biomni_eval_{model.replace('/', '_')}_{timestamp}_details.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved DETAILS JSON: {path}")


def maybe_print_details(run_cfg: EvalRunConfig, result_rec: dict, pred_map: Dict[str, str]):
    if not run_cfg.print_details:
        return

    status = result_rec["status"].lower()
    want = run_cfg.print_on.lower()
    if want not in {"fail", "pass", "all"}:
        want = "fail"

    counter_attr = "_printed_count"
    printed = getattr(maybe_print_details, counter_attr, 0)
    if printed >= run_cfg.print_limit:
        return

    if want == "all" or (want == "fail" and status == "fail") or (want == "pass" and status == "pass"):
        print("\n" + "=" * 80)
        print(f"[DETAIL] id={result_rec['id']} status={result_rec['status']} latency={result_rec.get('latency_sec', 0):.3f}s")
        print(f"Dataset : {result_rec['dataset_path']}")
        print(f"Question: {result_rec['question']}")
        print(f"Expected: {result_rec['expected']}")
        print(f"Raw Out : {result_rec['prediction_raw']}")
        print(f"Parsed  : {pred_map}")
        print(f"Matched : {result_rec['validation']['matched_keys']}")
        print(f"Missing : {result_rec['validation']['missing_keys']}")
        print(f"Mismatches: {result_rec['validation']['mismatches']}")
        print(f"Extra   : {result_rec['validation']['extra_keys']}")
        print(f"Feedback: {result_rec.get('feedback','')}")
        print("=" * 80 + "\n")
        setattr(maybe_print_details, counter_attr, printed + 1)


def should_collect_detail(run_cfg: EvalRunConfig, status: str, collected_count: int) -> bool:
    if not run_cfg.save_details:
        return False
    if collected_count >= run_cfg.details_limit:
        return False
    want = run_cfg.details_on.lower()
    status = status.lower()
    if want == "all":
        return True
    if want == "fail" and status == "fail":
        return True
    if want == "pass" and status == "pass":
        return True
    return False


def _finalize_and_exit(results, details_samples, out_dir, model, timestamp, run_cfg):
    save_main_reports(results, out_dir, model, timestamp)
    save_details_json(details_samples, out_dir, model, timestamp, run_cfg.details_on, run_cfg.details_limit)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate DA-Agent benchmark with Biomni agent.")
    parser.add_argument("--llm", type=str, required=True, help="LLM name for Biomni (e.g., 'claude-3-7-sonnet-20250219', 'gpt-4o').")
    parser.add_argument("--data-root", type=str, required=True, help="Path to ./benchmarks/DA-Agent")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all).")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in question list.")
    parser.add_argument("--sleep-seconds", type=int, default=0, help="Sleep between calls to handle rate limits.")
    parser.add_argument("--output-dir", type=str, default="evaluation_outputs", help="Where to save results.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle questions before evaluating.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for shuffling (0 = random).")
    # Dataset layout overrides
    parser.add_argument("--questions-fn", type=str, default="da-dev-questions.jsonl", help="Questions filename.")
    parser.add_argument("--labels-fn", type=str, default="da-dev-labels.jsonl", help="Labels filename.")
    parser.add_argument("--csv-subdir", type=str, default=os.path.join("data", "datasets", "csv"), help="Relative CSV subdir under data-root.")
    # Printing options (stdout)
    parser.add_argument("--print-details", action="store_true", help="Print raw outputs & parsed details to stdout.")
    parser.add_argument("--print-on", type=str, default="fail", choices=["fail", "pass", "all"], help="When to print details.")
    parser.add_argument("--print-limit", type=int, default=5, help="Max number of detailed prints.")
    # Save a capped subset of detailed samples to a separate JSON (like eval_infi.py feedback feature)
    parser.add_argument("--save-details", action="store_true", help="Save a subset of detailed samples to a separate *_details.json file.")
    parser.add_argument("--details-on", type=str, default="fail", choices=["fail", "pass", "all"], help="Which results to save in details JSON.")
    parser.add_argument("--details-limit", type=int, default=10, help="Max number of samples to include in details JSON.")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.output_dir).resolve()

    # Timestamp for filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_cfg = EvalRunConfig(
        timestamp=timestamp,
        print_details=bool(args.print_details),
        print_on=args.print_on,
        print_limit=int(args.print_limit),
        save_details=bool(args.save_details),
        details_on=args.details_on,
        details_limit=int(args.details_limit),
    )

    # Configure Biomni
    default_config.llm = args.llm
    # Make timeouts generous for heavy tasks
    if not getattr(default_config, "timeout_seconds", None):
        default_config.timeout_seconds = 1200
    else:
        default_config.timeout_seconds = max(default_config.timeout_seconds, 600)

    # Initialize agent (path to current working dir)
    agent = A1(path=str(Path.cwd()))

    # Load benchmark
    items, _ = load_benchmark(
        data_root,
        questions_fn=args.questions_fn,
        labels_fn=args.labels_fn,
        csv_subdir=args.csv_subdir,
    )
    logger.info(f"Loaded {len(items)} examples from {data_root}.")

    # Optional shuffle
    if args.shuffle:
        import random
        if args.seed != 0:
            random.seed(args.seed)
        random.shuffle(items)

    # Slicing
    if args.offset > 0:
        items = items[args.offset:]
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    results: List[dict] = []
    details_samples: List[dict] = []

    for idx, item in enumerate(tqdm(items, desc="Evaluating"), start=1):
        if _SHOULD_FLUSH_AND_EXIT:
            logger.warning("Early stop requested. Flushing partial results...")
            break

        if not item.dataset_path.exists():
            logger.error(f"[{idx}/{len(items)}] Dataset not found: {item.dataset_path}")
            result_error = {
                "id": item.id,
                "status": "ERROR",
                "question": item.question,
                "expected": item.expected_map,
                "prediction_raw": "",
                "dataset_path": str(item.dataset_path),
                "validation": {
                    "missing_keys": list(item.expected_map.keys()),
                    "mismatches": {},
                    "matched_keys": {},
                    "extra_keys": [],
                },
                "reason": "Dataset file not found",
                "latency_sec": 0.0,
                "feedback": "Dataset file not found at provided path.",
            }
            results.append(result_error)

            # Collect details if requested
            if should_collect_detail(run_cfg, result_error["status"], len(details_samples)):
                details_samples.append({
                    "id": item.id,
                    "status": "ERROR",
                    "question": item.question,
                    "dataset_path": str(item.dataset_path),
                    "prompt": build_prompt(item),
                    "raw_output": "",
                    "parsed_tags": {},
                    "expected": item.expected_map,
                    "matched_keys": {},
                    "missing_keys": list(item.expected_map.keys()),
                    "mismatches": {},
                    "extra_keys": [],
                    "latency_sec": 0.0,
                    "feedback": "Dataset file not found at provided path.",
                })

            maybe_print_details(run_cfg, result_error, {})
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            continue

        prompt = build_prompt(item)
        logger.info(f"[{idx}/{len(items)}] Running Biomni on dataset: {item.dataset_path}")
        t0 = time.time()
        try:
            resp = agent.go(prompt)
            text = coerce_output_to_text(resp).strip()
            latency = time.time() - t0
        except Exception as e:
            latency = time.time() - t0
            logger.exception(f"[{idx}/{len(items)}] Biomni crashed on item {item.id}: {e}")
            result_exc = {
                "id": item.id,
                "status": "ERROR",
                "question": item.question,
                "expected": item.expected_map,
                "prediction_raw": f"EXCEPTION: {type(e).__name__}: {e}",
                "dataset_path": str(item.dataset_path),
                "validation": {
                    "missing_keys": list(item.expected_map.keys()),
                    "mismatches": {},
                    "matched_keys": {},
                    "extra_keys": [],
                },
                "reason": str(e),
                "latency_sec": round(latency, 3),
                "feedback": f"Exception while calling agent: {type(e).__name__}: {e}",
            }
            results.append(result_exc)

            if should_collect_detail(run_cfg, result_exc["status"], len(details_samples)):
                details_samples.append({
                    "id": item.id,
                    "status": "ERROR",
                    "question": item.question,
                    "dataset_path": str(item.dataset_path),
                    "prompt": prompt,
                    "raw_output": result_exc["prediction_raw"],
                    "parsed_tags": {},
                    "expected": item.expected_map,
                    "matched_keys": {},
                    "missing_keys": list(item.expected_map.keys()),
                    "mismatches": {},
                    "extra_keys": [],
                    "latency_sec": round(latency, 3),
                    "feedback": result_exc["feedback"],
                })

            maybe_print_details(run_cfg, result_exc, {})
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
            continue

        pred_map = parse_predicted_tags(text)
        is_pass, matched, mismatches, missing, extra = evaluate_prediction(pred_map, item.expected_map)
        fb = feedback_text(pred_map, matched, mismatches, missing, extra)

        result_rec = {
            "id": item.id,
            "status": "PASS" if is_pass else "FAIL",
            "question": item.question,
            "expected": item.expected_map,
            "prediction_raw": text,
            "dataset_path": str(item.dataset_path),
            "validation": {
                "missing_keys": missing,
                "mismatches": mismatches,
                "matched_keys": matched,
                "extra_keys": extra,
            },
            "latency_sec": round(latency, 3),
            "feedback": fb,
        }
        results.append(result_rec)

        if should_collect_detail(run_cfg, result_rec["status"], len(details_samples)):
            details_samples.append({
                "id": item.id,
                "status": result_rec["status"],
                "question": item.question,
                "dataset_path": str(item.dataset_path),
                "prompt": prompt,
                "raw_output": text,
                "parsed_tags": pred_map,
                "expected": item.expected_map,
                "matched_keys": matched,
                "missing_keys": missing,
                "mismatches": mismatches,
                "extra_keys": extra,
                "latency_sec": round(latency, 3),
                "feedback": fb,
            })

        maybe_print_details(run_cfg, result_rec, pred_map)

        if args.sleep_seconds > 0 and idx < len(items):
            time.sleep(args.sleep_seconds)

    # Save files (also runs on early stop)
    _finalize_and_exit(results, details_samples, out_dir, args.llm, timestamp, run_cfg)


if __name__ == "__main__":
    main()
