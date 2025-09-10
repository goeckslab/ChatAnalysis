#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation runner for the DA-Agent benchmark (tags-only output) with optional verbose dumps,
the ability to re-run a subset of examples (e.g., those that failed due to rate limits),
and early .env loading for API keys.

New flags:
- --only-ids "1,2,3"                     # run only these IDs
- --only-ids-file path.json[|.jsonl]     # extract IDs from a prior results file
- --reason-substr "RateLimitError"       # when using --only-ids-file, filter details by reason substring

Timestamped outputs by default:
  results -> evaluation_outputs/results_<model>_<YYYYmmdd-HHMMSS>.json
  traces  -> evaluation_outputs/traces/traces_<model>_<YYYYmmdd-HHMMSS>.jsonl
Use --no-timestamp-out to disable timestamping for the results file if needed.

.env support:
- --env-file path/to/.env    # load this .env before reading env vars
- --no-env                   # skip loading any .env
If neither is provided, a default .env (if present) will be loaded without overriding real env vars.

Example:
  python eval_infi.py \
    --model openai/gpt-4o \
    --data-root ./benchmarks/DA-Agent \
    --compile-examples ./examples.json \
    --limit 10 \
    --sleep-seconds 60 \
    --dump-on both --dump-limit 10
"""

from __future__ import annotations

import os
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import re

# Optional dotenv (.env) loader
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # dotenv is optional

# Your DA agent utilities
from da import get_agent, get_compiled_agent, PythonCodeTool


ROOT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=ROOT_LOG_FORMAT, force=True)
logger = logging.getLogger("eval_infi")

QUESTIONS_FILE = "da-dev-questions.jsonl"
LABELS_FILE = "da-dev-labels.jsonl"
DATASET_SUBDIR = Path("data/datasets/csv")

TAG_RE = re.compile(r"@([A-Za-z0-9_]+)\[(.*?)\]")


# ---------------------------------------
# Helpers
# ---------------------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def read_json_any(path: Path) -> Any:
    txt = path.read_text(encoding="utf-8")
    # try plain JSON
    try:
        return json.loads(txt)
    except Exception:
        pass
    # try JSONL
    items = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    if items:
        return items
    raise ValueError(f"Unrecognized JSON/JSONL at {path}")


def format_ground_truth(common_answers: List[List[str]]) -> Dict[str, str]:
    # Convert [["mean_fare","34.65"], ...] -> {"mean_fare":"34.65", ...}
    return {str(k): str(v) for k, v in common_answers}


def build_context(dataset_abs: Path, constraints: str, fmt: str) -> str:
    return (
        f"Dataset absolute path: {dataset_abs}\n"
        f"Constraints: {constraints}\n"
        f"Expected format string: {fmt}\n"
        "ABSOLUTE EVAL RULES (TAGS-ONLY):\n"
        "- Compute values with the python_code_executor tool; do not guess.\n"
        "- Your FINAL OUTPUT must be exactly the Expected format string with placeholders replaced.\n"
        "- Preserve tag names, order, commas, and spacing exactly as given.\n"
        "- DO NOT include any extra tags or any extra text (no JSON, no explanations).\n"
        "Examples:\n"
        "  Expected: @mean_fare[mean_fare_value]            -> @mean_fare[34.65]\n"
        "  Expected: @a[x], @b[y] (keep comma+space)        -> @a[1.23], @b[4.56]\n"
        "\n"
        "FINAL OUTPUT: Output only one line containing the filled @tag[value] pairs, and nothing else.\n"
    )


def inject_dataset_path(agent, dataset_path: Path):
    """Ensure PythonCodeTool uses this dataset path for the example."""
    try:
        if hasattr(agent, "_ensure_tool"):
            t = agent._ensure_tool()
            if isinstance(t, PythonCodeTool):
                t.current_dataset_path = dataset_path
    except Exception:
        pass
    try:
        tools = getattr(getattr(agent, "react_agent", None), "tools", []) or []
        for t in tools:
            if isinstance(t, PythonCodeTool):
                t.current_dataset_path = dataset_path
    except Exception:
        pass


def extract_tags_from_prediction(pred_text: str) -> Dict[str, str]:
    """
    Accept either:
    - raw '@tag[value], @tag2[value2]' text, or
    - JSON with a 'benchmark_tags' field containing that string.
    """
    # Try JSON first
    tags_src = pred_text
    try:
        data = json.loads(pred_text)
        tags_src = data.get("benchmark_tags", pred_text)
    except Exception:
        pass

    found = {}
    for m in TAG_RE.finditer(tags_src):
        found[m.group(1)] = m.group(2)
    return found


class FocusLogCapture(logging.Handler):
    """
    Captures log lines that are most helpful to understand "how it solved it",
    especially the PythonCodeTool outputs. You can widen this filter if needed.
    """
    def __init__(self, level=logging.INFO):
        super().__init__(level)
        self._lines: List[str] = []
        self._fmt = logging.Formatter(ROOT_LOG_FORMAT)

    def emit(self, record: logging.LogRecord):
        msg = self._fmt.format(record)
        name = record.name or ""
        # keep PythonCodeTool lines and high-signal agent lines
        if ("PythonCodeTool" in msg) or ("PREFLIGHT" in msg) or ("ReAct" in name):
            self._lines.append(msg)

    def text(self, max_chars: int = 50000) -> str:
        out = "\n".join(self._lines)
        return out[:max_chars]


def should_dump(
    idx: int,
    status: str,
    dump_on: str,
    dump_limit: int,
    dump_every: int,
    pass_count: int,
    fail_count: int,
    pass_cap: int,
    fail_cap: int,
) -> bool:
    # Every Nth example
    if dump_every and ((idx + 1) % dump_every == 0):
        return True

    # By status
    if dump_on == "none":
        return False
    if dump_on == "pass" and status != "PASS":
        return False
    if dump_on == "fail" and status != "FAIL":
        return False
    if dump_on == "both" and status not in ("PASS", "FAIL"):
        return False

    # Per-status caps
    if status == "PASS" and pass_cap and pass_count >= pass_cap:
        return False
    if status == "FAIL" and fail_cap and fail_count >= fail_cap:
        return False

    return True


def numeric_close(pred: str, exp: str) -> bool:
    """Compare numeric strings with tolerance inferred from expected's decimals."""
    try:
        p = float(pred)
        e = float(exp)
        decs = len(exp.split(".")[1]) if "." in exp else 0
        tol = 0.5 * (10 ** (-decs))  # half-ULP at requested decimals
        return abs(p - e) <= tol + 1e-12
    except Exception:
        return False


def parse_only_ids_arg(ids_str: Optional[str]) -> Set[int]:
    if not ids_str:
        return set()
    out: Set[int] = set()
    for part in ids_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            logger.warning(f"Could not parse id '{part}' in --only-ids; skipping.")
    return out


def parse_ids_from_file(path: Path, reason_substr: Optional[str]) -> Set[int]:
    """
    Accepts:
    - prior results.json with {"summary":...,"details":[{id,...},...]}
    - a JSON array of detail objects
    - JSONL (one object per line)
    - *any* JSON where we can find objects with an 'id' (and optional 'reason')
    Applies reason_substr filter if provided (case-insensitive substring match on 'reason').
    """
    data = read_json_any(path)
    ids: Set[int] = set()

    def maybe_add(obj: Dict[str, Any]):
        if not isinstance(obj, dict):
            return
        if "id" not in obj:
            return
        if reason_substr:
            rs = str(obj.get("reason", ""))
            if reason_substr.lower() not in rs.lower():
                return
        try:
            ids.add(int(obj["id"]))
        except Exception:
            pass

    if isinstance(data, dict):
        if "details" in data and isinstance(data["details"], list):
            for row in data["details"]:
                maybe_add(row)
        else:
            for v in data.values():
                if isinstance(v, dict):
                    maybe_add(v)
                elif isinstance(v, list):
                    for x in v:
                        maybe_add(x)
    elif isinstance(data, list):
        for row in data:
            maybe_add(row)
    else:
        logger.warning("Unrecognized structure in --only-ids-file; no IDs extracted.")

    return ids


def run_eval(
    model: str,
    api_key: str,
    data_root: Path,
    examples_path: Path | None,
    limit: int,
    sleep_seconds: int,
    dump_on: str,
    dump_limit: int,
    dump_every: int,
    dump_pass_cap: int,
    dump_fail_cap: int,
    dump_dir: Path,
    only_ids: Set[int] | None,
) -> Dict[str, Any]:
    questions_path = data_root / QUESTIONS_FILE
    labels_path = data_root / LABELS_FILE
    questions = read_jsonl(questions_path)
    labels = read_jsonl(labels_path)
    labels_map = {row["id"]: row for row in labels}

    # optional filtering
    if only_ids:
        idset = set(only_ids)
        before = len(questions)
        questions = [q for q in questions if int(q.get("id")) in idset]
        missing = idset.difference({int(q.get("id")) for q in questions})
        if missing:
            logger.warning(
                f"--only-ids requested {len(idset)} ids, but {len(missing)} not found in questions: "
                f"{sorted(missing)[:10]}{' ...' if len(missing)>10 else ''}"
            )
        logger.info(f"Filtered questions from {before} -> {len(questions)} using --only-ids/--only-ids-file.")

    logger.info(f"Loaded {len(questions)} examples for this run.")

    outputs_dir = Path("evaluation_outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d-%H%M%S")
    dump_jsonl = dump_dir / f"traces_{model.replace('/', '_')}_{ts}.jsonl"
    dumped_total = 0
    dumped_pass = 0
    dumped_fail = 0

    if examples_path and examples_path.exists():
        logger.info(f"Compiling agent using examples: {examples_path.resolve()}")
        agent = get_compiled_agent(
            api_key=api_key,
            model_id_with_prefix=model,
            outputs_dir=outputs_dir,
            examples_file_path=examples_path,
            current_dataset_path=None,
        )
    else:
        agent = get_agent(
            api_key=api_key,
            model_id_with_prefix=model,
            outputs_dir=outputs_dir,
            current_dataset_path=None,
        )

    total = 0
    passed = 0
    details: List[Dict[str, Any]] = []

    if limit and limit > 0:
        questions = questions[:limit]

    N = len(questions)
    for idx, q in enumerate(questions):
        qid = q["id"]
        fmt = q.get("format", "")
        constraints = q.get("constraints", "")
        file_name = q["file_name"]
        dataset_abs = (data_root / DATASET_SUBDIR / file_name).resolve()

        if not dataset_abs.exists():
            rec = {
                "id": qid,
                "status": "SKIP",
                "reason": f"dataset not found: {dataset_abs}",
                "question": q["question"],
                "expected": fmt,
                "dataset_path": str(dataset_abs),
            }
            details.append(rec)
            continue

        label_row = labels_map.get(qid)
        if not label_row:
            rec = {
                "id": qid,
                "status": "SKIP",
                "reason": "no label found",
                "question": q["question"],
                "expected": fmt,
                "dataset_path": str(dataset_abs),
            }
            details.append(rec)
            continue

        expected_map = format_ground_truth(label_row.get("common_answers", []))
        context = build_context(dataset_abs, constraints, fmt)

        inject_dataset_path(agent, dataset_abs)

        logger.info(f"[{idx+1}/{N}] Using dataset: {dataset_abs}")

        cap = FocusLogCapture(level=logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(cap)

        try:
            pred = agent(question=q["question"], context=context)
            pred_text = pred.answer if hasattr(pred, "answer") else str(pred)
        except Exception as e:
            root_logger.removeHandler(cap)
            logs_text = cap.text()
            logger.exception(f"[{idx+1}/{N}] crashed.")
            rec = {
                "id": qid,
                "status": "ERROR",
                "reason": str(e),
                "question": q["question"],
                "expected": fmt,
                "dataset_path": str(dataset_abs),
            }
            details.append(rec)

            # dump trace on error if allowed
            if should_dump(
                idx, "FAIL", dump_on, dump_limit, dump_every, dumped_pass, dumped_fail, dump_pass_cap, dump_fail_cap
            ) and (dump_limit == 0 or dumped_total < dump_limit):
                dump_payload = {
                    "id": qid,
                    "status": "ERROR",
                    "question": q["question"],
                    "dataset_path": str(dataset_abs),
                    "context": context,
                    "prediction_raw": f"EXCEPTION: {type(e).__name__}: {e}",
                    "logs": logs_text,
                }
                with dump_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dump_payload) + "\n")
                dumped_total += 1
                dumped_fail += 1

            if sleep_seconds > 0 and idx + 1 < N:
                logger.info(f"Pacing for rate limits: sleeping {sleep_seconds}s...")
                time.sleep(sleep_seconds)
            continue
        finally:
            try:
                root_logger.removeHandler(cap)
            except Exception:
                pass

        logs_text = cap.text()
        predicted_tags = extract_tags_from_prediction(pred_text)

        # Validate
        missing_keys: List[str] = []
        mismatches: Dict[str, Dict[str, str]] = {}
        matched_keys: Dict[str, str] = {}
        for k, exp_val in expected_map.items():
            pred_val = predicted_tags.get(k)
            if pred_val is None:
                missing_keys.append(k)
            else:
                # numeric tolerant compare if looks numeric; else case-insensitive exact text
                is_num = False
                try:
                    float(exp_val)
                    float(pred_val)
                    is_num = True
                except Exception:
                    pass
                if is_num:
                    if numeric_close(pred_val, exp_val):
                        matched_keys[k] = pred_val
                    else:
                        mismatches[k] = {"expected": exp_val, "predicted": pred_val}
                else:
                    if str(pred_val).strip().lower() == str(exp_val).strip().lower():
                        matched_keys[k] = pred_val
                    else:
                        mismatches[k] = {"expected": exp_val, "predicted": pred_val}

        status = "PASS" if (len(missing_keys) == 0 and len(mismatches) == 0) else "FAIL"
        if status == "PASS":
            passed += 1
        total += 1

        rec = {
            "id": qid,
            "status": status,
            "question": q["question"],
            "expected": fmt,
            "prediction": pred_text,
            "dataset_path": str(dataset_abs),
            "validation": {
                "expected_tags": expected_map,
                "predicted_tags": predicted_tags,
                "missing_keys": missing_keys,
                "mismatches": mismatches,
                "matched_keys": matched_keys,
                "extra_keys": [k for k in predicted_tags.keys() if k not in expected_map],
            },
        }
        details.append(rec)

        # Optional dump (PASS/FAIL/both/every Nth)
        if should_dump(
            idx, status, dump_on, dump_limit, dump_every, dumped_pass, dumped_fail, dump_pass_cap, dump_fail_cap
        ) and (dump_limit == 0 or dumped_total < dump_limit):
            dump_payload = {
                "id": qid,
                "status": status,
                "question": q["question"],
                "dataset_path": str(dataset_abs),
                "context": context,
                "prediction_raw": pred_text,
                "predicted_tags": predicted_tags,
                "validation": rec["validation"],
                "logs": logs_text,
            }
            with dump_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(dump_payload) + "\n")
            dumped_total += 1
            if status == "PASS":
                dumped_pass += 1
            else:
                dumped_fail += 1

        if sleep_seconds > 0 and idx + 1 < N:
            logger.info(f"Pacing for rate limits: sleeping {sleep_seconds}s...")
            time.sleep(sleep_seconds)

    summary = {
        "total": total,
        "passed": passed,
        "accuracy": (passed / total) if total else 0.0,
        "model": model,
        "dumps_written": dumped_total,
        "dump_path": str(dump_jsonl),
        "filtered_by_ids": sorted(list(only_ids or [])),
    }
    return {"summary": summary, "details": details}


def main():
    # --- pre-parse env-related flags so we can load .env BEFORE reading defaults from os.getenv ---
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", type=Path, default=None,
                     help="Path to a .env file to load before reading API keys (optional).")
    pre.add_argument("--no-env", action="store_true",
                     help="Do not load any .env file (even if present).")
    pre_args, _ = pre.parse_known_args()

    # Load .env early so parser defaults that call os.getenv() can see the vars
    if not pre_args.no_env and load_dotenv is not None:
        if pre_args.env_file is not None:
            load_dotenv(dotenv_path=pre_args.env_file, override=False)
        else:
            # load default .env if present; no override of real env vars
            load_dotenv(override=False)

    # --- main parser (inherits the env flags so they appear in --help) ---
    parser = argparse.ArgumentParser(
        parents=[pre],
        description="DA-Agent benchmark evaluator with tag-only output, selective re-run, and verbose traces"
    )
    parser.add_argument("--model", required=True,
                        help="e.g., openai/gpt-4o, groq/llama3-70b-8192, google/gemini-2.5-pro")

    # NOTE: now that .env is loaded, defaults below will see variables from it
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY") or "",
        help="API key (if omitted, taken from OPENAI_API_KEY/GROQ_API_KEY/GEMINI_API_KEY)."
    )

    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--compile-examples", type=Path, default=None,
                        help="examples.json to compile few-shot behavior")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=int, default=0)

    # Output handling (timestamped by default)
    parser.add_argument("--out", type=Path, default=Path("evaluation_outputs/results.json"),
                        help="Path or filename for results. Timestamp will be appended unless --no-timestamp-out is set.")
    parser.add_argument("--no-timestamp-out", action="store_true",
                        help="Disable appending timestamp to results filename.")

    # Dump options
    parser.add_argument("--dump-on", choices=["none", "pass", "fail", "both"], default="both",
                        help="Which results to dump verbose traces for.")
    parser.add_argument("--dump-limit", type=int, default=10,
                        help="Max number of dumps to write (0 = unlimited).")
    parser.add_argument("--dump-every", type=int, default=0,
                        help="Additionally dump every Nth example (0 = disabled).")
    parser.add_argument("--dump-pass-cap", type=int, default=3,
                        help="Max PASS dumps (0 = unlimited).")
    parser.add_argument("--dump-fail-cap", type=int, default=7,
                        help="Max FAIL dumps (0 = unlimited).")
    parser.add_argument("--dump-dir", type=Path, default=Path("evaluation_outputs/traces"),
                        help="Directory to write JSONL dumps.")

    # Selection filters
    parser.add_argument("--only-ids", type=str, default="",
                        help="Comma-separated list of example IDs to run (e.g., '474,480,490').")
    parser.add_argument("--only-ids-file", type=Path, default=None,
                        help="Path to a prior results JSON/JSONL from which to extract example IDs.")
    parser.add_argument("--reason-substr", type=str, default="",
                        help="When using --only-ids-file, include only entries whose 'reason' contains this substring (case-insensitive).")

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Please provide --api-key or set OPENAI_API_KEY / GROQ_API_KEY / GEMINI_API_KEY (via env or .env).")

    # Build ID set (union of --only-ids and --only-ids-file filtered by --reason-substr)
    only_ids: Set[int] = set()
    only_ids |= parse_only_ids_arg(args.only_ids)
    if args.only_ids_file and args.only_ids_file.exists():
        rs = args.reason_substr.strip() or None
        from_file = parse_ids_from_file(args.only_ids_file, rs)
        only_ids |= from_file
        if not from_file and rs:
            logger.warning(f"--only-ids-file provided but no IDs matched reason substring '{rs}'.")

    # Run
    report = run_eval(
        model=args.model,
        api_key=args.api_key,
        data_root=args.data_root,
        examples_path=args.compile_examples,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        dump_on=args.dump_on,
        dump_limit=args.dump_limit,
        dump_every=args.dump_every,
        dump_pass_cap=args.dump_pass_cap,
        dump_fail_cap=args.dump_fail_cap,
        dump_dir=args.dump_dir,
        only_ids=only_ids,
    )

    # Results path (timestamped by default)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_timestamp_out:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_path.with_name(f"{out_path.stem}_{args.model.replace('/', '_')}_{ts}{out_path.suffix}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("==== SUMMARY ====")
    logger.info(json.dumps(report["summary"], indent=2))
    logger.info(f"Results written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
