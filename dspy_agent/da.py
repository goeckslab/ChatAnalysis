"""
DSPy data-analysis agent tailored for the DA-Agent benchmark (tags-only output).

Key behavior
- Exposes `dataset_path_in_tool_code` (Path) to model-run Python via a PythonCodeTool.
- Runs a CSV preflight if a dataset path exists (prints PREFLIGHT_OK).
- Forces the model to compute values with python_code_executor and then emit ONLY the
  required @tag[value] pairs exactly as specified by the benchmark's "format" string:
  * replace only the placeholders inside [...] with computed values
  * preserve tag names, order, commas, and spacing exactly
  * DO NOT output any extra text (no JSON, no explanations)

This file only contains the agent logic so that eval_infi.py can call it directly.
"""

from __future__ import annotations

import logging
import sys
from io import StringIO
from pathlib import Path
from functools import lru_cache
from typing import List, Optional
import re

import dspy
from dspy.teleprompt import BootstrapFewShot

try:
    import cloudpickle as pickle
except Exception:
    import pickle


AGENT_GENERATED_FILES_SUBDIR = Path("generated_files")

# Modules the tool allows user code to import
AUTHORIZED_MODULES_FOR_CODE_TOOL = [
    "pandas",
    "numpy",
    "matplotlib.pyplot",
    "seaborn",
    "scipy.stats",
    "sklearn",
    "statistics",
    "pathlib",
    "io",
    "random",
    "joblib",
    "openpyxl",
    "statsmodels",
    "plotly",
    "itertools",
    "collections",
    "json",
]


class PythonCodeTool(dspy.Tool):
    name = "python_code_executor"
    input_variable = "code"
    output_variable = "tool_output"
    description = (
        "Executes Python code for data analysis. Read CSVs from `dataset_path_in_tool_code` (a pathlib.Path). "
        "Save files under outputs_dir/'generated_files' and print their relative paths "
        "(e.g., 'generated_files/plot.png')."
    )

    def __init__(self, outputs_dir: Path, current_dataset_path: Optional[Path]):
        super().__init__(func=self.__call__)
        self.outputs_dir = Path(outputs_dir)
        self.agent_files_dir = self.outputs_dir / AGENT_GENERATED_FILES_SUBDIR
        self.agent_files_dir.mkdir(parents=True, exist_ok=True)
        self.current_dataset_path = Path(current_dataset_path) if current_dataset_path else None

    def __call__(self, code: str) -> str:
        logging.info(f"PythonCodeTool: executing code (first 160 chars): {code[:160]}...")
        import builtins as _builtins_module

        def _custom_safe_import(name, globals_map=None, locals_map=None, fromlist=(), level=0):
            top = name.split(".")[0]
            allowed = (
                top in AUTHORIZED_MODULES_FOR_CODE_TOOL
                or any(name == m or name.startswith(m + ".") for m in AUTHORIZED_MODULES_FOR_CODE_TOOL)
            )
            if allowed:
                return _builtins_module.__import__(name, globals_map, locals_map, fromlist, level)
            raise ImportError(
                f"Import of '{name}' is restricted. Allowed: {AUTHORIZED_MODULES_FOR_CODE_TOOL}"
            )

        restricted_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len, "abs": abs, "str": str,
                "int": int, "float": float, "list": list, "dict": dict, "set": set,
                "tuple": tuple, "zip": zip, "enumerate": enumerate, "sorted": sorted,
                "all": all, "any": any, "isinstance": isinstance, "open": open,
                "round": round, "sum": sum, "min": min, "max": max,
                "getattr": getattr, "hasattr": hasattr, "repr": repr, "callable": callable,
                "True": True, "False": False, "None": None,
                "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
                "IndexError": IndexError, "KeyError": KeyError, "AttributeError": AttributeError,
                "NameError": NameError, "FileNotFoundError": FileNotFoundError,
                "ImportError": ImportError, "RuntimeError": RuntimeError,
                "NotImplementedError": NotImplementedError, "ZeroDivisionError": ZeroDivisionError,
                "__import__": _custom_safe_import,
            },
            "outputs_dir": self.outputs_dir,
            "dataset_path_in_tool_code": self.current_dataset_path,
            "Path": Path,
        }

        # Pre-import allowlisted modules and provide aliases
        for mod in AUTHORIZED_MODULES_FOR_CODE_TOOL:
            try:
                if "." in mod:
                    parts = mod.split(".")
                    obj = _builtins_module.__import__(parts[0], fromlist=parts[1:])
                    for p in parts[1:]:
                        obj = getattr(obj, p)
                    alias = parts[-1]
                    restricted_globals[alias] = obj
                    if mod == "matplotlib.pyplot":
                        restricted_globals["plt"] = obj
                else:
                    restricted_globals[mod] = _builtins_module.__import__(mod)
            except ImportError as e:
                logging.warning(f"Could not pre-import authorized module '{mod}': {e}")

        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            p = self.current_dataset_path
            if p is None:
                print("DATASET_PRECHECK: path is None")
            else:
                try:
                    print(f"DATASET_PRECHECK: exists={p.exists()} path={p}")
                except Exception:
                    print(f"DATASET_PRECHECK: exists=? path={p}")

            exec(code, restricted_globals)
            stdout_val = captured.getvalue()
            logging.info(f"PythonCodeTool STDOUT (first 500): {stdout_val[:500]}")
            return f"STDOUT:\n{stdout_val}\nExecution successful."[:3000]
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            err = f"Execution failed.\nERROR_TYPE: {type(e).__name__}\nERROR_MESSAGE: {e}\nTRACEBACK:\n{tb}"
            logging.error(f"PythonCodeTool error: {err[:1000]}")
            return err[:3000]
        finally:
            sys.stdout = old_stdout


PREFLIGHT_CODE = """\
import pandas as pd
from pathlib import Path
p = dataset_path_in_tool_code
assert p is not None, "dataset_path_in_tool_code is None"
print(f"USING_DATASET_PATH={p}")
df = pd.read_csv(p)
print(f"LOADED_SHAPE={df.shape}")
print("PREFLIGHT_OK")
"""


class DataAnalysisSignature(dspy.Signature):
    """
    You are an expert data analysis assistant using ReAct.

    HARD REQUIREMENTS
    1) If a dataset path is available, FIRST call python_code_executor with the preflight code to
       load the CSV and print PREFLIGHT_OK.
    2) Then perform the analysis by calling python_code_executor again to compute the required values.
       DO NOT GUESS.
    3) Build the final output by taking the **Expected format string** from the context and replacing
       ONLY the placeholders inside square brackets with your computed values (preserving tag names,
       order, punctuation, and spacing). Use the required rounding.

    FINAL OUTPUT (STRICT): Output a single line containing ONLY the filled @tag[value] pairs.
    - Do NOT include any extra tags such as @concept, @level, or @concepts.
    - Do NOT include any additional text or JSON. Only the tag line, nothing else.
    """
    context = dspy.InputField(desc="Must include Dataset absolute path, Constraints, Expected format string.")
    question = dspy.InputField(desc="Task to perform.")
    answer = dspy.OutputField(desc="A single-line string with only the filled @tag[value] pairs (no extra text).")


class DataAnalysisAgentModule(dspy.Module):
    def __init__(self, outputs_dir: Path, current_dataset_path: Optional[Path], max_iters: int = 10):
        super().__init__()
        self._outputs_dir = Path(outputs_dir) if outputs_dir else Path("evaluation_outputs")
        self._current_dataset_path = Path(current_dataset_path) if current_dataset_path else None
        self.tool = PythonCodeTool(outputs_dir=self._outputs_dir, current_dataset_path=self._current_dataset_path)
        self.react_agent = dspy.ReAct(
            DataAnalysisSignature,
            tools=[self.tool],
            max_iters=max_iters,
        )

    def _ensure_tool(self) -> PythonCodeTool:
        if hasattr(self, "tool") and isinstance(self.tool, PythonCodeTool):
            return self.tool
        tools = getattr(getattr(self, "react_agent", None), "tools", []) or []
        for t in tools:
            if isinstance(t, PythonCodeTool):
                self.tool = t
                return t
        t = PythonCodeTool(outputs_dir=self._outputs_dir, current_dataset_path=self._current_dataset_path)
        try:
            self.react_agent.tools = [t] + list(tools)
        except Exception:
            pass
        self.tool = t
        return t

    @staticmethod
    def _extract_path_from_context(context: str) -> Optional[Path]:
        # Parse any *.csv path in context; pick the first that exists.
        for m in re.finditer(r"([A-Za-z0-9_\-./\\:]+\.csv)", context or ""):
            p = Path(m.group(1)).expanduser()
            if p.exists():
                return p
        return None

    def forward(self, question, context):
        tool = self._ensure_tool()

        # Fallback: parse dataset path from context if missing
        if getattr(tool, "current_dataset_path", None) is None:
            parsed = self._extract_path_from_context(context or "")
            if parsed is not None:
                tool.current_dataset_path = parsed
                logging.info(f"Parsed dataset path from context: {parsed}")

        # Very explicit, non-ambiguous policy block oriented to tags-only output
        strict_block = (
            "\n\n[STRICT BENCHMARK OUTPUT POLICY]\n"
            "- Compute with python_code_executor; do NOT guess.\n"
            "- Your final output must be exactly the Expected format string with placeholders replaced.\n"
            "- Preserve tag names, order, punctuation, commas, and spaces exactly.\n"
            "- Output ONLY the filled @tag[value] pairs (single line). No extra text. No JSON. No commentary.\n"
        )

        # Preflight if dataset path exists
        if getattr(tool, "current_dataset_path", None) is not None:
            preflight_out = tool(PREFLIGHT_CODE)
            augmented_context = context + strict_block + "\n[PRELIGHT TOOL STDOUT]\n" + preflight_out
        else:
            augmented_context = (
                context + strict_block
                + "\n[PRELIGHT TOOL STDOUT]\nDATASET_PRECHECK: path is None\nSKIPPING_PREFLIGHT_DUE_TO_MISSING_PATH\n"
            )

        return self.react_agent(question=question, context=augmented_context)


def _simple_training_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """
    Tolerant training metric: accept any non-empty string that contains at least one @tag[value].
    """
    try:
        text = prediction.answer
        if not isinstance(text, str) or not text.strip():
            return False
        return bool(re.search(r"@[A-Za-z0-9_]+\[.*?\]", text))
    except Exception:
        return False


def load_examples_from_json(json_file_path: Path) -> List[dspy.Example]:
    """
    Load optional examples (EDA, etc.). These are used only to compile prompts.
    The 'answer' field may be JSON or text; we don't enforce it during training.
    """
    import json
    items = []
    if not json_file_path or not Path(json_file_path).exists():
        logging.info("No examples file found; skipping training examples.")
        return items
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data:
        ex = dspy.Example(
            question=row.get("question"),
            context=row.get("context"),
            rationale=row.get("rationale"),
            answer=row.get("final_answer") or row.get("answer") or "",
        ).with_inputs("question", "context")
        items.append(ex)
    logging.info(f"Loaded {len(items)} training examples from {json_file_path}")
    return items


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)


def _ensure_tool_on_agent(agent, outputs_dir: Path, current_dataset_path: Optional[Path]):
    # Make sure a PythonCodeTool is attached, and we can overwrite its dataset path later.
    try:
        if hasattr(agent, "_ensure_tool"):
            return agent
        tools = getattr(getattr(agent, "react_agent", None), "tools", []) or []
        for t in tools:
            if isinstance(t, PythonCodeTool):
                setattr(agent, "tool", t)
                return agent
        t = PythonCodeTool(outputs_dir=outputs_dir, current_dataset_path=current_dataset_path)
        try:
            agent.react_agent.tools = [t] + list(tools)
        except Exception:
            pass
        setattr(agent, "tool", t)
    except Exception as e:
        logging.debug(f"_ensure_tool_on_agent failed: {e}")
    return agent


@lru_cache(maxsize=4)
def get_agent(
    api_key: str,
    model_id_with_prefix: str,
    outputs_dir: Path,
    current_dataset_path: Optional[Path],
):
    lm = dspy.LM(model_id_with_prefix, api_key=api_key)
    dspy.settings.configure(lm=lm, trace=None)
    agent = DataAnalysisAgentModule(outputs_dir=outputs_dir, current_dataset_path=current_dataset_path)
    return agent


def get_compiled_agent(
    api_key: str,
    model_id_with_prefix: str,
    outputs_dir: Path,
    examples_file_path: Path,
    current_dataset_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
):
    lm = dspy.LM(model_id_with_prefix, api_key=api_key)
    dspy.settings.configure(lm=lm, trace=None)

    cache_dir = cache_dir or (outputs_dir / "compiled_agents")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"compiled_{_slug(model_id_with_prefix)}.pkl"

    # Try to load compiled agent
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                compiled = pickle.load(f)
            logging.info(f"Loaded compiled agent from cache: {cache_path}")
            return _ensure_tool_on_agent(compiled, outputs_dir, current_dataset_path)
        except Exception as e:
            logging.warning(f"Failed to load cached agent ({cache_path}): {e}. Recompiling...")

    # Compile afresh
    base_agent = DataAnalysisAgentModule(outputs_dir=outputs_dir, current_dataset_path=current_dataset_path)
    examples = load_examples_from_json(examples_file_path)
    if not examples:
        logging.warning("No training examples found; returning uncompiled agent.")
        return base_agent

    teleprompter = BootstrapFewShot(
        metric=_simple_training_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=min(len(examples), 4),
    )
    try:
        compiled = teleprompter.compile(base_agent, trainset=examples)
        with open(cache_path, "wb") as f:
            pickle.dump(compiled, f)
        logging.info(f"Compiled agent saved to {cache_path}")
        return _ensure_tool_on_agent(compiled, outputs_dir, current_dataset_path)
    except Exception as e:
        logging.warning(f"Compilation failed, using uncompiled agent: {e}")
        return base_agent
