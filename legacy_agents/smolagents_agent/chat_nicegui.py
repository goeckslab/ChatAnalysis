import os
import re
import base64
import pandas as pd
from collections import deque
from dotenv import load_dotenv
import json
import uuid
import logging
import sys
from pathlib import Path
import psycopg2
import asyncio
import argparse # Ensure this is at the top

import yaml
import importlib.resources

# NiceGUI imports
from nicegui import ui, app, Client
from nicegui.events import UploadEventArguments
from nicegui import app
from functools import lru_cache

from smolagents_agent.prompt import CODE_AGENT_SYSTEM_PROMPT

# Smolagents imports
try:
    from smolagents import CodeAgent, LiteLLMModel
except ImportError:
    logging.error("smolagents or LiteLLMModel not found. Ensure it's installed and in PYTHONPATH.")
    CodeAgent = object
    LiteLLMModel = object

# --- Global Constants and Configuration ---
OPENAI_API_KEY_FILE = Path("user_config_openai.key")
GROQ_API_KEY_FILE = Path("user_config_groq.key")
DEFAULT_OUTPUT_DIR = Path("outputs_dir")
DEFAULT_CHAT_HISTORY_FILE = Path("chat_history_nicegui.json")
SCRIPT_PATH = Path(__file__).resolve().parent

load_dotenv()
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def get_custom_prompt_templates() -> dict: # Or PromptTemplates if imported
    try:
        # Load the default prompt templates YAML from the smolagents library
        default_prompts_yaml = importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
        custom_templates = yaml.safe_load(default_prompts_yaml)
    except Exception as e:
        logging.error(f"Could not load default smolagents prompts. Using a basic structure. Error: {e}")
        # Fallback if default loading fails (ensure all required keys for PromptTemplates are present)
        from smolagents.agents import EMPTY_PROMPT_TEMPLATES # Adjust path if needed
        custom_templates = EMPTY_PROMPT_TEMPLATES.copy() # Make a copy

    # Replace the system_prompt with your custom one
    custom_templates["system_prompt"] = CODE_AGENT_SYSTEM_PROMPT

    # Ensure other necessary keys from PromptTemplates are present if using EMPTY_PROMPT_TEMPLATES as fallback
    # For example, if EMPTY_PROMPT_TEMPLATES doesn't have all nested dicts:
    if "planning" not in custom_templates:
        custom_templates["planning"] = {"initial_plan": "", "update_plan_pre_messages": "", "update_plan_post_messages": ""}
    if "managed_agent" not in custom_templates:
        custom_templates["managed_agent"] = {"task": "", "report": ""}
    if "final_answer" not in custom_templates:
        custom_templates["final_answer"] = {"pre_messages": "", "post_messages": ""}

    return custom_templates

# Prepare it once
CUSTOM_PROMPT_TEMPLATES = get_custom_prompt_templates()

# --- Helper Functions ---
def save_key_to_specific_file(file_path: Path, key_value: str):
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f: f.write(key_value)
        logging.info(f"API key saved to {file_path}")
    except Exception as e: logging.error(f"Error saving API key to {file_path}: {e}", exc_info=True)

def load_key_from_specific_file(file_path: Path) -> str | None:
    try:
        if file_path.exists():
            with open(file_path, "r") as f: key = f.read().strip()
            if key: logging.info(f"API key loaded from {file_path}"); return key
    except Exception as e: logging.error(f"Error loading API key from {file_path}: {e}", exc_info=True)
    return None

def check_db_env_vars():
    required_vars = ["PG_HOST_DA", "PG_DB_DA", "PG_USER_DA", "PG_PASSWORD_DA"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars: logging.warning(f"Missing DB env vars: {missing_vars}"); return False
    return True

def get_db_connection():
    if not check_db_env_vars(): return None
    try:
        return psycopg2.connect(
            host=os.environ["PG_HOST_DA"], database=os.environ["PG_DB_DA"],
            user=os.environ["PG_USER_DA"], password=os.environ["PG_PASSWORD_DA"]
        )
    except Exception as e: logging.error(f"DB connection failed: {e}"); return None

def init_feedback_db():
    if not all(os.environ.get(var) for var in ["PG_HOST_DA", "PG_DB_DA", "PG_USER_DA", "PG_PASSWORD_DA"]):
        logging.warning("PostgreSQL environment variables not fully set. Feedback DB will not be initialized.")
        return False
    conn = get_db_connection()
    if not conn: logging.error("Cannot init feedback DB: No connection."); return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS message_feedback (
                    id SERIAL PRIMARY KEY, user_id TEXT NOT NULL, question TEXT NOT NULL,
                    answer TEXT NOT NULL, feedback TEXT NOT NULL, comment TEXT,
                    dataset_path TEXT, timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP);
            """)
            conn.commit()
        logging.info("Feedback DB initialized."); return True
    except Exception as e: logging.error(f"Error initializing feedback DB table: {e}", exc_info=True); return False
    finally:
        if conn: conn.close()

@lru_cache(maxsize=5)
def create_agent_cached(api_key, model_id_with_prefix):
    if LiteLLMModel is object or CodeAgent is object:
        raise RuntimeError("LiteLLMModel or CodeAgent not available. Smolagents might not be installed or failed to import.")
    model = LiteLLMModel(model_id=model_id_with_prefix, api_key=api_key)
    return CodeAgent(
        tools=[], model=model,
        prompt_templates=CUSTOM_PROMPT_TEMPLATES,
        additional_authorized_imports=[
            "pandas", "numpy", "matplotlib", "seaborn", "scipy", "sklearn",
            "pycaret", "plotly", "joblib", "io", "xgboost", "lightgbm",
            "catboost", "anndata", "Bio", "pysam", "gffutils"
        ], max_steps=20)

class NiceGuiApp:
    MODEL_OPTIONS_SELECT = {
        "gpt-4o": "OpenAI (GPT-4o)", "gpt-4o-mini": "OpenAI (GPT-4o-mini)",
        "gpt-4": "OpenAI (GPT-4)", "gpt-3.5-turbo": "OpenAI (GPT-3.5-Turbo)",
        "llama-3.3-70b-versatile": "Groq (Llama-3.3-70B)",
        "llama3-70b-8192": "Groq (Llama3-70B-8192)",
        "mixtral-8x7b-32768": "Groq (Mixtral-8x7B)",
    }

    def __init__(self, user_id: str, cli_args_ns: argparse.Namespace):
        self.user_id = user_id
        self.cli_args = cli_args_ns
        self.agent = None
        self.output_dir = Path(self.cli_args.generate_file_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.messages = []
        self.memory = deque(maxlen=30)
        self.bookmarks = [] # For bookmarking functionality
        
        self.current_dataset_file_path: Path | None = None
        self.current_dataset_display_name = "No dataset loaded"
        self.current_input_data_type = self.cli_args.input_data_type
        self.current_data_object = None
        self.summary_stats_csv_path: Path | None = None
        self.eda_report_path: Path | None = None
        self.db_available = False
        
        self.openai_api_key = ""
        self.groq_api_key = ""
        self.selected_model_id = "gpt-4o"
        self.selected_model_name = self.MODEL_OPTIONS_SELECT.get(self.selected_model_id, self.selected_model_id)
        
        self.selected_message_for_details_idx: int | None = None
        self.selected_bookmark_for_details: dict | None = None # To distinguish bookmark details from live chat details

        # UI element references
        self.chat_container: ui.column | None = None
        self.dataset_preview_area: ui.column | None = None
        self.sidebar_api_status_label: ui.label | None = None
        self.details_container: ui.column | None = None
        self.openai_key_input: ui.input | None = None
        self.groq_key_input: ui.input | None = None
        self.model_select_element: ui.select | None = None
        self.chat_input_field: ui.input | None = None
        self.left_drawer: ui.left_drawer | None = None
        self.bookmarks_container: ui.column | None = None # For sidebar bookmarks

        self.chat_history_file_path = Path(self.cli_args.chat_history_path)
        self.initial_dataset_path_from_arg: Path | None = Path(self.cli_args.input_file_path) if self.cli_args.input_file_path else None
        
        self.load_initial_state()

    def load_initial_state(self):
        cli_openai_path_str = self.cli_args.cli_openai_key_file_path
        if cli_openai_path_str:
            cli_openai_key = load_key_from_specific_file(Path(cli_openai_path_str))
            if cli_openai_key: self.openai_api_key = cli_openai_key; save_key_to_specific_file(OPENAI_API_KEY_FILE, cli_openai_key)
        if not self.openai_api_key: self.openai_api_key = load_key_from_specific_file(OPENAI_API_KEY_FILE) or ""

        cli_groq_path_str = self.cli_args.cli_groq_key_file_path
        if cli_groq_path_str:
            cli_groq_key = load_key_from_specific_file(Path(cli_groq_path_str))
            if cli_groq_key: self.groq_api_key = cli_groq_key; save_key_to_specific_file(GROQ_API_KEY_FILE, cli_groq_key)
        if not self.groq_api_key: self.groq_api_key = load_key_from_specific_file(GROQ_API_KEY_FILE) or ""

        if self.chat_history_file_path.exists():
            try:
                with open(self.chat_history_file_path, "r") as f: history = json.load(f)
                self.messages = history.get("messages", [])
                self.memory = deque(history.get("memory", []), maxlen=30)
                self.bookmarks = history.get("bookmarks", []) # Load bookmarks

                saved_dataset_path_str = history.get("analysis_file_path")
                if saved_dataset_path_str:
                    saved_dataset_path = Path(saved_dataset_path_str)
                    if saved_dataset_path.exists():
                        self.current_dataset_file_path = saved_dataset_path
                        self.current_dataset_display_name = self.current_dataset_file_path.name
                        self.current_input_data_type = history.get("input_data_type", self.current_input_data_type)
                
                if self.initial_dataset_path_from_arg and self.initial_dataset_path_from_arg.exists():
                     self.current_dataset_file_path = self.initial_dataset_path_from_arg
                     self.current_dataset_display_name = self.current_dataset_file_path.name
                     self.current_input_data_type = self.cli_args.input_data_type # Prioritize CLI type if file is from CLI

                summary_path_str = history.get("summary_stats_csv_path"); eda_path_str = history.get("eda_report_path")
                if summary_path_str and Path(summary_path_str).exists(): self.summary_stats_csv_path = Path(summary_path_str)
                if eda_path_str and Path(eda_path_str).exists(): self.eda_report_path = Path(eda_path_str)
                
                # Update 'bookmarked' status in self.messages based on loaded bookmarks
                bookmarked_message_timestamps = {bm.get('assistant_response', {}).get('timestamp') for bm in self.bookmarks if bm.get('assistant_response')}
                for msg in self.messages:
                    if msg.get("role") == "assistant" and msg.get("timestamp") in bookmarked_message_timestamps:
                        msg['bookmarked'] = True
                
                logging.info(f"Chat history loaded from {self.chat_history_file_path}")
            except Exception as e: logging.error(f"Error loading chat history: {e}", exc_info=True)
        self.db_available = init_feedback_db()

    def save_chat_history(self):
        history = {
            "messages": self.messages, "memory": list(self.memory),
            "bookmarks": self.bookmarks, # Save bookmarks
            "analysis_file_path": str(self.current_dataset_file_path) if self.current_dataset_file_path else None,
            "input_data_type": self.current_input_data_type,
            "summary_stats_csv_path": str(self.summary_stats_csv_path) if self.summary_stats_csv_path else None,
            "eda_report_path": str(self.eda_report_path) if self.eda_report_path else None,
        }
        try:
            with open(self.chat_history_file_path, "w") as f: json.dump(history, f, indent=2)
            logging.info(f"Chat history (including bookmarks) saved to {self.chat_history_file_path}.")
        except Exception as e: logging.error(f"Error saving chat history: {e}", exc_info=True)

    def _get_final_api_key_and_model_id(self):
        is_openai = self.selected_model_id.startswith("gpt-")
        is_groq = any(kw in self.selected_model_id for kw in ["llama", "mixtral", "gemma"])
        final_api_key, final_model_id_for_agent = None, self.selected_model_id
        if is_openai: final_api_key = self.openai_api_key
        elif is_groq:
            final_api_key = self.groq_api_key
            if not final_model_id_for_agent.startswith("groq/"): final_model_id_for_agent = "groq/" + final_model_id_for_agent
        return final_api_key, final_model_id_for_agent

    def try_initialize_agent(self):
        final_api_key, final_model_id_for_agent = self._get_final_api_key_and_model_id()
        status_message, status_color = "", ""
        logging.info(f"Attempting to initialize agent with model: {final_model_id_for_agent}. API Key present: {bool(final_api_key)}")
        if final_api_key and final_model_id_for_agent:
            try:
                self.agent = create_agent_cached(final_api_key, final_model_id_for_agent)
                logging.info(f"system prompt: {self.agent.system_prompt}")
                status_message = f"Agent Ready ({self.selected_model_name})"
                status_color = 'green'; ui.notify(status_message, type='positive', timeout=3000, position='top'); logging.info(status_message)
            except Exception as e:
                self.agent = None; error_str = str(e).lower()
                logging.error(f"Agent init failed for model '{final_model_id_for_agent}' using key '***{final_api_key[-4:] if final_api_key and len(final_api_key) > 4 else 'EMPTY/SHORT'}': {e}", exc_info=True)
                auth_keywords = ["authentication", "api key", "invalid key", "permission denied", "unauthorized", "401"]
                model_not_found_keywords = ["model_not_found", "does not exist", "404"]
                if any(keyword in error_str for keyword in auth_keywords) or isinstance(e, getattr(sys.modules.get("litellm.exceptions", object), "AuthenticationError", tuple())):
                    status_message = f"Agent Error: API Key for {self.selected_model_name} seems invalid or lacks permissions. Verify key. (Details in server console)"
                elif any(keyword in error_str for keyword in model_not_found_keywords) or isinstance(e, getattr(sys.modules.get("litellm.exceptions", object), "ModelNotFound", tuple())):
                    status_message = f"Agent Error: Model '{self.selected_model_name}' not found/accessible. Check model name & key. (Details in server console)"
                else: status_message = f"Agent Error: Failed to initialize {self.selected_model_name}. Check server console."
                ui.notify(status_message + f" Error: {str(e)[:100]}...", type='negative', multi_line=True, classes='w-96 whitespace-pre-wrap', auto_close=False, position='center')
                status_color = 'red'
        else:
            self.agent = None; missing_parts = []
            provider_name = "the selected provider"
            if self.selected_model_id:
                if self.selected_model_id.startswith("gpt-"): provider_name = "OpenAI"
                elif any(kw in self.selected_model_id for kw in ["llama", "mixtral", "gemma"]): provider_name = "Groq"
            if not final_api_key: missing_parts.append(f"API Key missing for {provider_name}")
            if not final_model_id_for_agent or final_model_id_for_agent == "groq/": missing_parts.append("a valid model is not selected")
            status_message = "Agent Not Ready: " + (" and ".join(missing_parts) if missing_parts else "Unknown configuration issue.") + ". Configure in sidebar."
            status_color = 'orange'; ui.notify(status_message, type='warning', multi_line=True, classes='w-96 whitespace-pre-wrap', auto_close=False, position='center')
            logging.warning(status_message)
        if self.sidebar_api_status_label:
            self.sidebar_api_status_label.set_text(status_message)
            self.sidebar_api_status_label.style(f'color: {status_color}; font-weight: bold; font-size: 0.8rem;')
            self.sidebar_api_status_label.tooltip(status_message if len(status_message) > 40 else '')
        return self.agent is not None

    async def handle_upload(self, e: UploadEventArguments):
        if not e.content: ui.notify("No file content.", type='negative'); return
        uploaded_filename = e.name
        self.current_input_data_type = Path(uploaded_filename).suffix.lower().replace('.', '') or 'csv'
        temp_file_path = self.output_dir / uploaded_filename
        try:
            with open(temp_file_path, 'wb') as f: f.write(e.content.read())
            self.current_dataset_file_path = temp_file_path; self.current_dataset_display_name = uploaded_filename
            ui.notify(f"File '{uploaded_filename}' processed.", type='positive')
            self.summary_stats_csv_path = self.eda_report_path = None
            self.messages.append({
                "role": "system", 
                "content": f"New dataset loaded: {uploaded_filename}. Please analyze.", 
                "type": "text", 
                "timestamp": pd.Timestamp.now(tz='UTC').isoformat() 
            })
            self.update_chat_display(); await self.preview_loaded_or_uploaded_dataset(); self.save_chat_history()
        except Exception as ex:
            ui.notify(f"Error processing '{uploaded_filename}': {ex}", type='negative', multi_line=True); logging.error(f"Upload error for {uploaded_filename}: {ex}", exc_info=True)
            self.current_dataset_file_path = None; self.current_dataset_display_name = "No dataset"

    def load_data_object_from_path(self, file_path: Path, data_type: str):
        try:
            if data_type == 'csv': return pd.read_csv(file_path)
            elif data_type == "tsv": return pd.read_csv(file_path, sep="\t")
            elif data_type == "h5ad": import anndata; return anndata.read_h5ad(file_path)
            elif data_type in ("xlsx", "xls"): return pd.read_excel(file_path)
            elif data_type == "json": return pd.read_json(file_path)
            elif data_type == "parquet": return pd.read_parquet(file_path)
            elif data_type == "h5": return pd.read_hdf(file_path)
            elif data_type in ("fa", "fasta"): from Bio import SeqIO; return list(SeqIO.parse(str(file_path), "fasta"))
            elif data_type == "vcf": import pysam; return pysam.VariantFile(str(file_path))
            elif data_type in ("gtf", "gff"):
                import gffutils; db_path = self.output_dir / f"{file_path.stem}.{data_type}.db"
                return gffutils.create_db(str(file_path), dbfn=str(db_path), force=True, keep_order=True, merge_strategy="merge", sort_attribute_values=True)
            elif data_type == "bed": return pd.read_csv(file_path, sep="\t", header=None)
            else: raise ValueError(f"Unsupported file type for direct load: {data_type}")
        except ImportError as ie: ui.notify(f"Missing library for {data_type}: {ie}. Install it.", type='error', multi_line=True, auto_close=False); logging.error(f"ImportError loading {file_path.name} ({data_type}): {ie}", exc_info=True); return None
        except Exception as e: ui.notify(f"Error loading {file_path.name} ({data_type}): {e}", type='negative', multi_line=True, auto_close=False); logging.error(f"Load error for {file_path} ({data_type}): {e}", exc_info=True); return None

    async def preview_loaded_or_uploaded_dataset(self):
        if not self.current_dataset_file_path or not self.dataset_preview_area:
            if self.dataset_preview_area: self.dataset_preview_area.clear();
            with self.dataset_preview_area: ui.label("No dataset selected.")
            return
        self.dataset_preview_area.clear()
        with self.dataset_preview_area:
            ui.label(f"Active: {self.current_dataset_display_name} ({self.current_input_data_type.upper()})").classes('text-md font-semibold mb-1')
            self.current_data_object = self.load_data_object_from_path(self.current_dataset_file_path, self.current_input_data_type)
            if self.current_data_object is None: ui.label("Failed to load data for preview.").classes('text-red-500'); return
            preview_table_height_classes = 'h-[200px] max-h-[200px] overflow-auto w-full bordered'
            summary_table_height_classes = 'h-[280px] max-h-[280px] overflow-auto w-full bordered'
            props_for_table = 'dense flat bordered separator=cell'
            if isinstance(self.current_data_object, pd.DataFrame):
                ui.markdown("###### Data Preview (Top 5)"); ui.table.from_pandas(self.current_data_object.head(5)).classes(preview_table_height_classes).props(props_for_table)
                if not self.summary_stats_csv_path or not self.summary_stats_csv_path.exists(): self.summary_stats_csv_path = self.generate_and_save_pandas_summary_csv(self.current_data_object)
                if self.summary_stats_csv_path and self.summary_stats_csv_path.exists():
                    try:
                        summary_df = pd.read_csv(self.summary_stats_csv_path, index_col=0)
                        ui.markdown("###### Summary Statistics").classes('mt-2'); ui.table.from_pandas(summary_df).classes(summary_table_height_classes).props(props_for_table)
                        ui.button("Download Summary", icon="download", on_click=lambda: ui.download(str(self.summary_stats_csv_path), filename=self.summary_stats_csv_path.name)).props("dense size=sm flat").classes("mt-1 text-sm text-indigo-600 hover:text-indigo-800")
                    except Exception as e: ui.notify(f"Err displaying summary: {e}", type='warning'); logging.warning(f"Summary CSV err {self.summary_stats_csv_path}: {e}")
            elif hasattr(self.current_data_object, "obs") and hasattr(self.current_data_object, "var"): # AnnData
                ui.markdown("###### AnnData Obs (Top 5)"); ui.table.from_pandas(self.current_data_object.obs.head(5)).classes(preview_table_height_classes).props(props_for_table)
                ui.markdown("###### AnnData Vars (Top 5)").classes('mt-2'); ui.table.from_pandas(self.current_data_object.var.head(5)).classes(preview_table_height_classes).props(props_for_table)
            elif isinstance(self.current_data_object, list) and self.current_input_data_type in ("fa", "fasta"): # FASTA
                ui.markdown("###### FASTA (First 3)")
                for i, record in enumerate(self.current_data_object[:3]): ui.markdown(f"**ID:** `{record.id}`\n**Seq (60bp):** `{str(record.seq)[:60]}...`").classes("text-xs")
            elif hasattr(self.current_data_object, "header") and self.current_input_data_type == "vcf": # VCF (pysam)
                ui.markdown("###### VCF Header"); ui.code(str(self.current_data_object.header)).classes('max-h-40 overflow-auto text-xs')
                ui.markdown("###### VCF Records (First 3)").classes("mt-1"); recs = list(self.current_data_object.fetch(max_records=3));
                for rec in recs: ui.code(str(rec)).classes("text-xs")
            elif hasattr(self.current_data_object, "all_features") and self.current_input_data_type in ("gtf", "gff"): # gffutils DB
                ui.markdown("###### GTF/GFF Features (First 3)").classes("mt-1"); features = list(self.current_data_object.all_features(limit=3))
                for feature in features: ui.code(str(feature)).classes("text-xs")
            else: ui.label(f"Enhanced preview for {self.current_input_data_type.upper()} not fully shown here.")

    def generate_and_save_pandas_summary_csv(self, dataframe: pd.DataFrame) -> Path | None:
        if not isinstance(dataframe, pd.DataFrame): return None
        original_filename_stem = Path(self.current_dataset_display_name).stem if self.current_dataset_display_name and self.current_dataset_display_name != "No dataset loaded" else "dataset"
        try:
            summary_df = dataframe.describe(include='all')
            summary_filename = f"summary_stats_for_{original_filename_stem}_{uuid.uuid4().hex[:6]}.csv"
            summary_csv_path = self.output_dir / summary_filename
            summary_df.to_csv(summary_csv_path, index=True)
            logging.info(f"Pandas summary saved: {summary_csv_path}")
            return summary_csv_path
        except Exception as e: logging.error(f"Error gen/save pandas summary: {e}", exc_info=True); ui.notify(f"Error in summary stats: {e}", type='negative'); return None

    def get_agent_prompt_for_nicegui(self, user_question: str, question_type: int = 2):
        memory_history_str = "\n".join(self.memory)
        memory_context = f"PREVIOUS CONVERSATION HISTORY (for context only, do not repeat yourself):\n{memory_history_str}\nEND OF PREVIOUS CONVERSATION.\n\n" if memory_history_str else "This is the beginning of the conversation.\n\n"
        # core_instructions = (
        #     "You are an expert data analysis assistant who can solve any task by writing and executing Python code.\n"
        #     "To solve the task, you MUST plan step-by-step and respond by thinking, then writing a single Python code block, then waiting for an observation. This cycle repeats: Thought, Code, Observation.\n"
        #     "Authorized libraries include pandas, numpy, matplotlib, seaborn, scipy, sklearn, pycaret, plotly, joblib, io, xgboost, lightgbm, catboost, anndata, Bio, pysam, gffutils.\n"
        # )
        dataset_info = (
            f"The primary dataset is located at: '{self.current_dataset_file_path}'.\n"
            f"This dataset is of type: '{self.current_input_data_type}'. Use this type to determine how to read/load it. Ignore the file extension if it conflicts.\n"
            f"Any plots or data files you generate MUST be saved to this relative directory: '{self.output_dir}'.\n"
            "Ensure filenames are descriptive and unique (e.g., append a random suffix like `_1a2b3c.png`). Do NOT use generic placeholders like 'XXXX'.\n"
        )
        output_format_instructions = (
             "   Always call the final_answer tool, providing the final answer in the following dictionary format (do not format as a JSON code block):\n"
                '{ "explanation": ["Your explanation here, in plain text. This can include detailed information or step-by-step guidance."], '
                '"plots": ["<path_to_the_image>" (leave the list empty if no plots are needed)], '
                '"files": ["<path_to_the_file>" (leave the list empty if no files are needed)], '
                '"next_steps_suggestion": ["List of possible next questions the user could ask to gain further insights. They should be questions. Only include this when the user has not explicitly asked for suggestions."] }'
        )
        current_task_prompt = ""
        if question_type == 0:
            current_task_prompt = (f"Perform a comprehensive Exploratory Data Analysis (EDA) on the provided dataset. Specifically address: {user_question}\n Always generate plots and supporting data files where appropriate for an EDA.")
        elif question_type == 1:
             current_task_prompt = f"Summarize the previous conversation provided in the history concisely. User's request: {user_question}"
        else: 
            current_task_prompt = (f"Address the following user question: {user_question}\nBefore answering, critically analyze if the question is multifaceted, ambiguous, or covers several distinct aspects. If so, provide three distinct candidate solutions using the 'candidate_solutions' JSON format. Otherwise, if the question is straightforward, provide a single concise answer using the standard JSON format. Most questions should be straightforward.")
        final_prompt = (f"{memory_context}\n{dataset_info}\nCURRENT TASK: {current_task_prompt}\n\nOUTPUT REQUIREMENTS:\n{output_format_instructions}\nRemember to think step-by-step using the Thought, Code, Observation cycle before calling final_answer().")
        return final_prompt

    def parse_response_content_for_nicegui(self, content_str: str | dict ):
        if isinstance(content_str, dict): return content_str
        if not isinstance(content_str, str):
            logging.warning(f"Agent response not string or dict: {type(content_str)}. Content: {str(content_str)[:100]}")
            return {"explanation": f"Unexpected response format: {str(content_str)[:100]}", "plots": [], "files": [], "next_steps_suggestion": []}
        try:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content_str, re.DOTALL)
            json_str = match.group(1).strip() if match else content_str.strip()
            try: parsed = json.loads(json_str)
            except json.JSONDecodeError:
                brace_start, brace_end = json_str.find('{'), json_str.rfind('}')
                if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                    potential_json = json_str[brace_start : brace_end+1]
                    try: parsed = json.loads(potential_json)
                    except json.JSONDecodeError as e_inner: logging.error(f"Robust JSON decode error: {e_inner}. String part: {potential_json[:200]}...", exc_info=True); raise e_inner
                else: logging.error(f"No clear JSON object in string: {json_str[:200]}..."); raise json.JSONDecodeError("No JSON delimiters", json_str, 0)
            if not isinstance(parsed.get("explanation") if "explanation" in parsed else parsed.get("candidate_solutions"), (str, list)):
                 if not ("candidate_solutions" in parsed and isinstance(parsed["candidate_solutions"], list)):
                    logging.warning(f"Parsed JSON unexpected structure: {str(parsed)[:200]}")
                    return {"explanation": f"Agent response (parsed, structure differs): {str(parsed)}", "plots": [], "files": [], "next_steps_suggestion": []}
            return parsed
        except Exception as e:
            logging.error(f"Error parsing agent response: {e}. Content: {content_str[:500]}", exc_info=True)
            return {"explanation": f"Could not parse LLM response. Raw content (first 200 chars): {content_str[:200]}", "plots": [], "files": [], "next_steps_suggestion": []}

    def update_sidebar_bookmarks(self):
        if not self.bookmarks_container:
            logging.warning("Bookmarks container not initialized, cannot update sidebar bookmarks.")
            return
        
        self.bookmarks_container.clear()
        with self.bookmarks_container:
            if not self.bookmarks:
                ui.label("No bookmarks yet.").classes("text-xs text-gray-500 p-2 text-center")
            else:
                for idx, bookmark in enumerate(self.bookmarks):
                    # Ensure essential keys exist, provide defaults if not
                    user_q = bookmark.get("user_question", "Bookmarked Item")
                    assistant_resp = bookmark.get("assistant_response", {})
                    assistant_content_snippet = str(assistant_resp.get("content", ""))[:70] + "..."

                    with ui.card().tight().classes("w-full my-1 shadow-md hover:shadow-lg transition-shadow cursor-pointer"):
                        with ui.card_section().classes("p-2"):
                            with ui.row().classes("w-full items-center justify-between no-wrap"):
                                with ui.column().classes("flex-grow").on('click', lambda b=bookmark: self.show_bookmark_details(b)): # Make text clickable
                                    ui.label(f"Q: {user_q[:50]}...").classes("text-xs font-semibold text-indigo-700").style("white-space: normal; word-break: break-word; line-height: 1.2;")
                                    ui.label(f"A: {assistant_content_snippet}").classes("text-xs text-gray-600 mt-1").style("white-space: normal; word-break: break-word; line-height: 1.2;")
                                ui.button(icon='delete_sweep', on_click=lambda i=idx: self.delete_bookmark(i), color='red-5') \
                                    .props('flat round dense size=xs').tooltip("Delete bookmark")
                        # Optional: Add timestamp or other small info if available in bookmark_data
                        # bookmark_timestamp = assistant_resp.get("timestamp")
                        # if bookmark_timestamp:
                        #     try: stamp_display = pd.to_datetime(bookmark_timestamp).strftime('%b %d, %I:%M %p')
                        #     except: stamp_display = str(bookmark_timestamp)[:16]
                        #     ui.label(stamp_display).classes("text-right text-xs text-gray-400 pr-2 pb-1")


    # ADD THIS METHOD:
    def delete_bookmark(self, bookmark_idx: int):
        if 0 <= bookmark_idx < len(self.bookmarks):
            deleted_bookmark_content = self.bookmarks[bookmark_idx].get('assistant_response', {}).get('content')
            timestamp_of_deleted_bookmark = self.bookmarks[bookmark_idx].get('assistant_response', {}).get('timestamp')
            
            del self.bookmarks[bookmark_idx]
            
            # Try to find and unmark the original message in the chat history
            # This relies on content and timestamp matching, which is okay but not perfectly robust.
            # Unique message IDs would be better if you implement them later.
            for msg in self.messages:
                if msg.get("role") == "assistant" and \
                   msg.get("content") == deleted_bookmark_content and \
                   msg.get("timestamp") == timestamp_of_deleted_bookmark:
                    if 'bookmarked' in msg:
                        del msg['bookmarked'] # Remove bookmarked flag
                    break 

            ui.notify("Bookmark removed.", type='info')
            self.save_chat_history()
            self.update_sidebar_bookmarks() # Refresh the list in the sidebar
            self.update_chat_display() # Refresh chat to update bookmark button states
            
            # If the deleted bookmark was being shown in details, clear details
            if self.selected_bookmark_for_details and \
               self.selected_bookmark_for_details.get('assistant_response', {}).get('content') == deleted_bookmark_content and \
               self.selected_bookmark_for_details.get('assistant_response', {}).get('timestamp') == timestamp_of_deleted_bookmark:
                self.selected_bookmark_for_details = None
                self.update_details_pane() # Show placeholder or default view in details
        else:
            ui.notify("Could not delete bookmark (invalid index).", type='negative')

    # Ensure show_details_for_message and show_bookmark_details are also present
    # (They were in my previous full code response and your logs indicate show_details_for_message was missing,
    # so I'm re-including it here for completeness if you missed it)

    # ADDED/CORRECTED show_details_for_message method
    def show_details_for_message(self, message_idx: int):
        self.selected_message_for_details_idx = message_idx
        self.selected_bookmark_for_details = None # Clear bookmark selection when viewing live message
        if self.details_container:
            self.update_details_pane()
        else:
            logging.warning("Attempted to show message details, but details_container is not yet initialized.")

    def show_bookmark_details(self, bookmark_data: dict):
        self.selected_bookmark_for_details = bookmark_data
        self.selected_message_for_details_idx = None # Clear live message selection
        if self.details_container:
            self.update_details_pane()
        else:
            logging.warning("Attempted to show bookmark details, but details_container is not yet initialized.")

    def add_bookmark(self, message_idx: int):
        if 0 <= message_idx < len(self.messages) and self.messages[message_idx].get("role") == "assistant":
            assistant_msg = self.messages[message_idx]
            if assistant_msg.get('bookmarked'):
                ui.notify("This response is already bookmarked.", type='info'); return

            user_question = "Context not found"
            for i in range(message_idx - 1, -1, -1):
                if self.messages[i].get("role") == "user":
                    user_question = self.messages[i].get("content", "User query not found"); break
            
            bookmark_data = {
                "user_question": user_question,
                "assistant_response": {
                    "content": assistant_msg.get("content"),
                    "plots": assistant_msg.get("plots", []),
                    "files": assistant_msg.get("files", []),
                    "middle_steps": assistant_msg.get("middle_steps"),
                    "timestamp": assistant_msg.get("timestamp") # Crucial for unmarking later
                }
            }
            self.bookmarks.append(bookmark_data)
            self.messages[message_idx]['bookmarked'] = True
            ui.notify("Response bookmarked!", type='positive')
            self.save_chat_history()
            self.update_sidebar_bookmarks()
            self.update_chat_display()
        else:
            ui.notify("Could not bookmark this message.", type='negative')

    def format_raw_middle_steps_for_display(self, raw_steps: list | None) -> str:
        if not raw_steps:
            return "*No intermediate steps were recorded by the agent.*"
        logging.info(f"Formatting {len(raw_steps)} middle steps.")
        formatted_steps_md_parts = ["#### Agent's Workings (Intermediate Steps)\n"]
        for idx, step_data in enumerate(raw_steps):
            step_number = step_data.get("step", idx + 1)
            current_step_md_parts = [f"##### Step {step_number}"]
            thought_content = step_data.get("thought", step_data.get("model_output", step_data.get("reasoning")))
            if thought_content:
                thought_str = str(thought_content).strip()
                task_marker_heading = "CURRENT TASK:" # Remove if agent echoes it
                if idx == 0 and task_marker_heading in thought_str:
                    lines = thought_str.splitlines()
                    filtered_lines = [line for line in lines if not line.strip().startswith(task_marker_heading) and not line.strip().startswith("Address the following user question:")]
                    thought_str = "\n".join(filtered_lines).strip() if filtered_lines else "(Agent processed task and planned)"
                    if not thought_str.strip(): thought_str = "(Agent's initial planning step)"
                current_step_md_parts.append(f"**Thought/Plan:**\n```text\n{thought_str}\n```")
            action = step_data.get("action"); action_input = step_data.get("action_input")
            if action:
                action_input_str = json.dumps(action_input, indent=2, default=str) if isinstance(action_input, dict) else str(action_input)
                current_step_md_parts.append(f"**Action Called:** `{action}`\n**Action Input:**\n```json\n{action_input_str}\n```")
            code_generated = step_data.get("code", step_data.get("code_generated"))
            if code_generated: current_step_md_parts.append(f"**Code Executed:**\n```python\n{str(code_generated).strip()}\n```")
            observation = step_data.get("observation", step_data.get("action_output", step_data.get("tool_outputs")))
            if observation is not None:
                obs_str = str(observation); obs_str = (obs_str[:700] + "\n... (observation truncated)") if len(obs_str) > 700 else obs_str
                current_step_md_parts.append(f"**Observation/Result:**\n```text\n{obs_str.strip()}\n```")
            error = step_data.get("error")
            if error: current_step_md_parts.append(f"**Error Encountered:**\n```text\n{str(error).strip()}\n```")
            if len(current_step_md_parts) > 1: formatted_steps_md_parts.append("\n\n".join(current_step_md_parts))
        return "\n\n---\n".join(formatted_steps_md_parts) if len(formatted_steps_md_parts) > 1 else "*No processable intermediate steps found.*"

    # async def handle_user_input(self, user_question: str | None):
    #     if not user_question or not user_question.strip():
    #         if self.chat_input_field: self.chat_input_field.set_value(None);
    #         return
    #     if self.chat_input_field: self.chat_input_field.set_value(None)
    #     if not self.agent: ui.notify("Agent not initialized...", type='warning', position='center', classes="p-4 text-lg", auto_close=False); return
    #     if not self.current_dataset_file_path: ui.notify("Upload dataset first.", type='warning', position='center'); return

    #     self.messages.append({"role": "user", "content": user_question, "type": "text", "timestamp": pd.Timestamp.now().isoformat()})
    #     self.memory.append(f"User: {user_question}")
    #     self.update_chat_display()

    #     thinking_indicator_row = None
    #     if self.chat_container:
    #         with self.chat_container:
    #             with ui.row().classes('w-full justify-center my-2') as thinking_indicator_row_instance: ui.spinner(size='lg', color='primary')
    #             thinking_indicator_row = thinking_indicator_row_instance
    #     formatted_middle_steps = "*Agent did not record detailed steps.*"
    #     try:
    #         prompt = self.get_agent_prompt_for_nicegui(user_question)
    #         logging.debug(f"Agent prompt for '{user_question[:30]}...': ...{prompt[-300:]}")
    #         if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'): self.agent.memory.clear(); logging.info("Agent memory cleared.")
    #         response_content = await asyncio.to_thread(self.agent.run, prompt)
    #         logging.info(f"Agent raw response: {str(response_content)[:500]}...")
    #         if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'get_full_steps'):
    #             raw_steps = self.agent.memory.get_full_steps(); logging.info(f"Raw middle steps (count: {len(raw_steps)}): {str(raw_steps)[:500]}...")
    #             formatted_middle_steps = self.format_raw_middle_steps_for_display(raw_steps)
    #         else: logging.warning("Agent memory/get_full_steps not found."); formatted_middle_steps = "*Agent config lacks detailed step providing.*"

    #         parsed_response = self.parse_response_content_for_nicegui(response_content)
    #         new_assistant_message_idx = -1
    #         if parsed_response:
    #             msg_to_append = {"role": "assistant", "timestamp": pd.Timestamp.now().isoformat(), "middle_steps": formatted_middle_steps}
    #             if "candidate_solutions" in parsed_response: msg_to_append.update({"content": "Agent proposed multiple approaches:", "type": "text_with_candidates", "candidates": parsed_response["candidate_solutions"], "next_steps": parsed_response.get("next_steps_suggestion", [])})
                
    #             else:
    #                 explanation_value = parsed_response.get("explanation", "Agent processed request.")
    #                 final_explanation_content_str: str
    #                 if isinstance(explanation_value, list):
    #                     # Join list elements into a single string. map(str, ...) ensures all elements are strings before joining.
    #                     final_explanation_content_str = "\n".join(map(str, explanation_value))
    #                 else:
    #                     # Value is already a string or some other type that needs to be stringified.
    #                     final_explanation_content_str = str(explanation_value)
                    
    #                 msg_to_append.update({
    #                     "content": final_explanation_content_str, # Use the processed string
    #                     "type": "text_with_attachments",
    #                     "plots": parsed_response.get("plots", []),
    #                     "files": parsed_response.get("files", []),
    #                     "next_steps": parsed_response.get("next_steps_suggestion", [])
    #                 })
    #             self.messages.append(msg_to_append); new_assistant_message_idx = len(self.messages) - 1
    #             self.memory.append(f"Assistant: {parsed_response.get('explanation', 'Response generated.')[:200]}...")
    #         else:
    #             self.messages.append({"role": "assistant", "content": f"Unprocessable response. Raw: {str(response_content)[:300]}...", "type": "error", "timestamp": pd.Timestamp.now().isoformat(), "middle_steps": formatted_middle_steps})
    #             self.memory.append(f"Assistant: Unprocessable response.")
    #         if new_assistant_message_idx != -1 and (self.messages[new_assistant_message_idx].get("plots") or self.messages[new_assistant_message_idx].get("files") or (self.messages[new_assistant_message_idx].get("middle_steps") and self.messages[new_assistant_message_idx]["middle_steps"].startswith("####"))):
    #             self.show_details_for_message(new_assistant_message_idx)
    #     except Exception as e:
    #         logging.error(f"Error during agent interaction for '{user_question}': {e}", exc_info=True)
    #         self.messages.append({"role": "assistant", "content": f"Internal error: {e}", "type": "error", "timestamp": pd.Timestamp.now().isoformat(), "middle_steps": "*Error during execution.*"})
    #     finally:
    #         if thinking_indicator_row: thinking_indicator_row.delete()
    #     self.update_chat_display(); self.save_chat_history()

    async def handle_user_input(self, user_question: str | None):
        if not user_question or not user_question.strip():
            ui.notify("Please enter a question.", type='warning')
            if self.chat_input_field:
                self.chat_input_field.set_value(None)
            return
        
        if self.chat_input_field:
            self.chat_input_field.set_value(None)

        # ---- MODIFIED CHECK: Use self.current_data_object instead of self.df ----
        is_load_command = any(keyword in user_question.lower() for keyword in ["load", "upload", "dataset", "file", "open"])
        if self.current_data_object is None and not is_load_command:
            ui.notify("Please upload or specify a dataset first, or ask to load one.", type='warning', position='center')
            return
        # ---- END MODIFIED CHECK ----

        self.messages.append({
            "role": "user", 
            "content": user_question, 
            "type": "text", 
            "timestamp": pd.Timestamp.now(tz='UTC').isoformat()
        })
        self.memory.append(f"User: {user_question}")
        self.update_chat_display()

        spinner_row_to_delete = None
        if self.chat_container:
            with self.chat_container:
                with ui.row().classes('w-full justify-center my-2') as temp_spinner_row:
                    ui.spinner(size='lg', color='primary')
                spinner_row_to_delete = temp_spinner_row

        try:
            if not self.agent:
                ui.notify("Agent not initialized. Please check API keys and model selection in the sidebar.", type='error', position='center', auto_close=False)
                if spinner_row_to_delete:
                    spinner_row_to_delete.delete()
                    spinner_row_to_delete = None
                return
            
            prompt = self.get_agent_prompt_for_nicegui(user_question)
            logging.debug(f"Agent prompt for '{user_question[:30]}...': ...{prompt[-300:]}")
            
            if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'clear'):
                self.agent.memory.clear()
                logging.info("Agent's internal memory cleared before run.")
                
            response_content = await asyncio.to_thread(self.agent.run, prompt)
            logging.info(f"Agent raw response: {str(response_content)[:500]}...")

            formatted_middle_steps = "*Agent did not provide detailed steps or steps could not be retrieved.*"
            if hasattr(self.agent, 'memory') and hasattr(self.agent.memory, 'get_full_steps'):
                raw_steps = self.agent.memory.get_full_steps()
                logging.info(f"Raw middle steps (count: {len(raw_steps)}): {str(raw_steps)[:500]}...")
                formatted_middle_steps = self.format_raw_middle_steps_for_display(raw_steps)
            else:
                logging.warning("Agent does not have 'memory.get_full_steps()' method to retrieve detailed steps.")
            
            parsed_response = self.parse_response_content_for_nicegui(response_content)
            new_assistant_message_idx = -1
            if parsed_response:
                msg_to_append = {
                    "role": "assistant",
                    "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
                    "middle_steps": formatted_middle_steps
                }
                # Using a more generic way to get primary text content from agent for memory
                primary_agent_text = "Agent processed request."
                if "explanation" in parsed_response: # This is for text_with_attachments
                    primary_agent_text = parsed_response["explanation"]
                elif "final_answer" in parsed_response: # This might be for text_with_candidates or other structures
                    primary_agent_text = parsed_response["final_answer"]
                elif isinstance(response_content, str): # Fallback for very simple responses
                    primary_agent_text = response_content

                if "candidate_solutions" in parsed_response:
                    msg_to_append.update({
                        "type": "text_with_candidates",
                        # Use a clearer key for main content if candidates are present
                        "content": parsed_response.get("overview", "Please review the options below."), 
                        "candidates": parsed_response.get("candidate_solutions", []),
                        "next_steps": parsed_response.get("next_steps_suggestion", [])
                    })
                else:
                    msg_to_append.update({
                        "type": "text_with_attachments",
                        "content": parsed_response.get("explanation", "Agent processed request."),
                        "plots": parsed_response.get("plots", []),
                        "files": parsed_response.get("files", []),
                        "next_steps": parsed_response.get("next_steps_suggestion", [])
                    })
                self.messages.append(msg_to_append)
                new_assistant_message_idx = len(self.messages) - 1
                self.memory.append(f"Agent: {str(primary_agent_text)[:200]}...")
                
                if new_assistant_message_idx != -1 and \
                (self.messages[new_assistant_message_idx].get("plots") or \
                    self.messages[new_assistant_message_idx].get("files") or \
                    (self.messages[new_assistant_message_idx].get("middle_steps") and \
                    self.messages[new_assistant_message_idx]["middle_steps"].startswith("####"))):
                    self.show_details_for_message(new_assistant_message_idx)
            else:
                self.messages.append({
                    "role": "assistant", 
                    "content": f"Unprocessable response from agent. Raw (first 300 chars): {str(response_content)[:300]}...", 
                    "type": "error", 
                    "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
                    "middle_steps": formatted_middle_steps
                })
                self.memory.append("Agent: [Unprocessable response]")

        except Exception as e:
            logging.error(f"Error during agent interaction for '{user_question}': {e}", exc_info=True)
            self.messages.append({
                "role": "assistant", 
                "content": f"An internal error occurred: {e}", 
                "type": "error", 
                "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
                "middle_steps": "*Error during agent execution phase.*"
            })
        finally:
            if spinner_row_to_delete:
                spinner_row_to_delete.delete()
                # spinner_row_to_delete = None # Not strictly necessary but good form
        
        self.update_chat_display()
        self.save_chat_history()

    # def update_details_pane(self): # Modified
    #     if not self.details_container: logging.warning("Details container NA for update."); return
    #     self.details_container.clear()

    #     source_data = None
    #     data_origin = None # 'live_message' or 'bookmark'

    #     if self.selected_bookmark_for_details:
    #         source_data = self.selected_bookmark_for_details
    #         data_origin = 'bookmark'
    #     elif self.selected_message_for_details_idx is not None and self.selected_message_for_details_idx < len(self.messages):
    #         source_data = self.messages[self.selected_message_for_details_idx]
    #         data_origin = 'live_message'
    #     elif self.selected_message_for_details_idx is None: # Default to last assistant message with details
    #         for i in range(len(self.messages) - 1, -1, -1):
    #             msg = self.messages[i]
    #             if msg.get("role") == "assistant" and (msg.get("plots") or msg.get("files") or (msg.get("middle_steps") and msg.get("middle_steps").startswith("####"))):
    #                 source_data = msg; data_origin = 'live_message_default'; break
        
    #     with self.details_container:
    #         if not source_data:
    #             ui.label("Select 'View Plots & Tables' or a bookmark for details.").classes("text-gray-500 m-4 text-center italic"); return
            
    #         if data_origin == 'live_message' or data_origin == 'live_message_default':
    #             msg_data = source_data
    #             if msg_data.get("role") != "assistant": ui.label("Details only for assistant messages.").classes("text-gray-500 m-4"); return
    #             ui.markdown(f"##### Agent Response Details")
    #             user_query = "N/A"; current_idx = self.messages.index(msg_data) if msg_data in self.messages else -1
    #             if current_idx > 0:
    #                 idx = current_idx -1
    #                 while idx >= 0: 
    #                     if self.messages[idx].get("role") == "user": user_query = self.messages[idx].get("content", "N/A"); break
    #                     idx -=1
    #             if user_query != "N/A":
    #                 # ui.markdown("###### Regarding Query:").classes("text-gray-700 font-semibold mt-1 text-sm")
    #                 ui.markdown(f"Query: {user_query[:250]}{'...' if len(user_query)>250 else ''}").classes("text-gray-800 p-2 text-sm bg-slate-100 rounded-md border")
                
    #             # Explanation is in main chat, plots, files, middle_steps are primary for details
    #             plots = msg_data.get("plots", []); files = msg_data.get("files", []); middle_steps_str = msg_data.get("middle_steps")
            
    #         elif data_origin == 'bookmark':
    #             ui.markdown("##### Bookmarked Item Details")
    #             user_query = source_data.get("user_question", "N/A")
    #             assistant_resp = source_data.get("assistant_response", {})
    #             # explanation = assistant_resp.get("content", "") # Not showing main explanation again
    #             plots = assistant_resp.get("plots", []); files = assistant_resp.get("files", []); middle_steps_str = assistant_resp.get("middle_steps")
    #             if user_query != "N/A":
    #                 ui.markdown("###### User Query:").classes("text-gray-700 font-semibold mt-1 text-sm")
    #                 ui.markdown(f"{user_query}").classes("text-gray-800 p-2 text-sm bg-slate-100 rounded-md border")

    #         ui.separator().classes("my-3")

    #         if middle_steps_str and middle_steps_str.strip() and middle_steps_str.startswith("####"):
    #             with ui.expansion("Agent's Workings (Intermediate Steps)", icon="list_alt", value=True).classes("w-full my-2 border rounded-md shadow-sm"): # Open by default
    #                 with ui.card_section().classes("bg-gray-50 p-2"):
    #                      ui.markdown(middle_steps_str).classes('middle-steps-content') # Uses CSS class
    #         elif middle_steps_str:
    #              ui.markdown("###### Agent's Workings:").classes("mt-2 text-sm")
    #              ui.markdown(middle_steps_str).classes("italic text-sm text-gray-500")

    #         if plots: # ... (Plot rendering logic from previous response)
    #             ui.markdown("###### Plots").classes("mt-2 text-sm")
    #             with ui.grid(columns=1).classes("gap-2 w-full"):
    #                 for plot_path_str in plots:
    #                     plot_path = Path(plot_path_str)
    #                     if plot_path.is_file():
    #                         with ui.card().tight().classes("w-full shadow-lg"):
    #                             ui.image(str(plot_path)).classes('max-w-full h-auto rounded-t-lg border-b object-contain')
    #                             with ui.card_actions().props("align=right"): ui.button(icon="download", on_click=lambda p=str(plot_path): ui.download(p)).props("flat dense size=sm color=primary round").tooltip("Download Plot")
    #                     else: ui.label(f"Plot file not found: {plot_path.name}").classes('text-red-400 text-xs')
    #         if files: # ... (File rendering logic from previous response)
    #             ui.markdown("###### Files & Data").classes("mt-2 text-sm")
    #             for file_path_str in files:
    #                 file_path = Path(file_path_str)
    #                 if file_path.is_file():
    #                     with ui.card().tight().classes("my-2 w-full shadow-lg"):
    #                         with ui.card_section().classes("flex justify-between items-center p-2"):
    #                             ui.label(file_path.name).classes("font-semibold text-sm"); ui.button(icon="download", on_click=lambda p=str(file_path): ui.download(p)).props("flat dense size=sm color=primary round").tooltip("Download File")
    #                         if file_path.suffix.lower() in ['.csv', '.tsv']:  # This is likely your line 585
    #             # IMPORTANT: The following lines MUST be indented
    #                             ui.separator()
    #                             with ui.card_section().classes("p-0"):
    #                                 try:
    #                                     df_preview = pd.read_csv(file_path, sep=',' if file_path.suffix.lower() == '.csv' else '\t')
    #                                     ui.table.from_pandas(df_preview.head(5)).classes('h-[200px] overflow-auto w-full bordered text-xs').props('dense flat bordered separator=cell')
    #                                 except Exception as e_df: 
    #                                     ui.label(f"Preview failed: {e_df}").classes('text-orange-400 text-xs p-2')
                            
    #                         elif file_path.suffix.lower() == '.html': # This is likely your line 586
    #                             # IMPORTANT: The following lines MUST be indented
    #                             ui.separator()
    #                             with ui.card_section():
    #                                 try:
    #                                     with open(file_path, 'r', encoding='utf-8') as f_html: 
    #                                         html_content = f_html.read()
    #                                     ui.html(html_content).classes('max-h-96 h-[350px] overflow-auto border w-full p-1')
    #                                 except Exception as e_html: 
    #                                     ui.label(f"HTML display failed: {e_html}").classes('text-orange-400 text-xs p-2')
            
    #         if not plots and not files and not (middle_steps_str and middle_steps_str.startswith("####")):
    #             ui.label("No specific plots, files, or detailed agent steps for this message.").classes("m-2 text-gray-500")

    # Inside your NiceGuiApp class:

    def update_details_pane(self):
        if not self.details_container:
            logging.warning("Details container NA for update.")
            return
        self.details_container.clear()

        source_data = None
        data_origin = None 

        # Logic to determine source_data and data_origin (from live message or bookmark)
        # This part should remain as you have it. For example:
        if self.selected_bookmark_for_details:
            source_data = self.selected_bookmark_for_details
            data_origin = 'bookmark'
        elif self.selected_message_for_details_idx is not None and \
             self.selected_message_for_details_idx < len(self.messages):
            source_data = self.messages[self.selected_message_for_details_idx]
            data_origin = 'live_message'
        elif self.selected_message_for_details_idx is None: # Default to last assistant message with details
            for i in range(len(self.messages) - 1, -1, -1):
                msg = self.messages[i]
                if msg.get("role") == "assistant" and \
                   (msg.get("plots") or msg.get("files") or \
                    (msg.get("middle_steps") and msg.get("middle_steps").startswith("####"))):
                    source_data = msg
                    data_origin = 'live_message_default'
                    break
        
        with self.details_container:
            if not source_data:
                ui.label("Select 'View Plots & Tables' or a bookmark for details.")\
                    .classes("text-gray-500 m-4 text-center italic")
                return
            
            # Initialize raw lists
            plots_raw = []
            files_raw = []
            middle_steps_str = None
            user_query = "N/A" # Default user query

            # Extract data based on origin (live message or bookmark)
            if data_origin == 'live_message' or data_origin == 'live_message_default':
                msg_data = source_data # source_data is the message dict
                # ... (code to display "Agent Response Details" and find user_query - keep this part) ...
                if msg_data.get("role") != "assistant": # Should be redundant if logic above is correct
                    ui.label("Details only for assistant messages.").classes("text-gray-500 m-4"); return
                ui.markdown(f"##### Agent Response Details")
                current_idx = self.messages.index(msg_data) if msg_data in self.messages else -1
                if current_idx > 0:
                    idx = current_idx -1
                    while idx >= 0: 
                        if self.messages[idx].get("role") == "user": user_query = self.messages[idx].get("content", "N/A"); break
                        idx -=1
                if user_query != "N/A":
                    ui.markdown("###### Regarding Query:").classes("text-gray-700 font-semibold mt-1 text-sm")
                    ui.markdown(f"{user_query[:250]}{'...' if len(user_query)>250 else ''}").classes("text-gray-800 p-2 text-sm bg-slate-100 rounded-md border")

                plots_raw = msg_data.get("plots", [])
                files_raw = msg_data.get("files", [])
                middle_steps_str = msg_data.get("middle_steps")
            
            elif data_origin == 'bookmark':
                ui.markdown("##### Bookmarked Item Details")
                user_query = source_data.get("user_question", "N/A")
                assistant_resp = source_data.get("assistant_response", {})
                if user_query != "N/A":
                    ui.markdown("###### User Query:").classes("text-gray-700 font-semibold mt-1 text-sm")
                    ui.markdown(f"{user_query}").classes("text-gray-800 p-2 text-sm bg-slate-100 rounded-md border")
                
                plots_raw = assistant_resp.get("plots", [])
                files_raw = assistant_resp.get("files", [])
                middle_steps_str = assistant_resp.get("middle_steps")

            ui.separator().classes("my-3")

            # Display Middle Steps (as you have it)
            if middle_steps_str and middle_steps_str.strip() and middle_steps_str.startswith("####"):
                with ui.expansion("Agent's Workings (Intermediate Steps)", icon="list_alt", value=True)\
                        .classes("w-full my-2 border rounded-md shadow-sm"):
                    with ui.card_section().classes("bg-gray-50 p-2"):
                        ui.markdown(middle_steps_str).classes('middle-steps-content')
            elif middle_steps_str:
                ui.markdown("###### Agent's Workings:").classes("mt-2 text-sm")
                ui.markdown(middle_steps_str).classes("italic text-sm text-gray-500")

            # --- FLATTEN AND DISPLAY PLOTS ---
            plots_flattened = []
            if plots_raw: # Check if plots_raw is not empty
                for item in plots_raw:
                    if isinstance(item, list): # If item is a list, iterate its sub-items
                        for sub_item in item:
                            if isinstance(sub_item, (str, Path)): # Ensure sub_item is a path-like string
                                plots_flattened.append(sub_item)
                            else:
                                logging.warning(f"Skipping invalid sub-item in plot path list: {sub_item} (type: {type(sub_item)})")
                    elif isinstance(item, (str, Path)): # If item is already a path-like string
                        plots_flattened.append(item)
                    else:
                        logging.warning(f"Skipping invalid item in plot path list: {item} (type: {type(item)})")
            
            if plots_flattened:
                ui.markdown("###### Plots").classes("mt-2 text-sm")
                with ui.grid(columns=1).classes("gap-2 w-full"): # Adjust columns if you want multiple plots per row
                    for plot_path_str in plots_flattened:
                        try:
                            plot_path = Path(plot_path_str) # This should now receive a string or Path object
                            if plot_path.is_file():
                                with ui.card().tight().classes("w-full shadow-lg"):
                                    ui.image(str(plot_path)).classes('max-w-full h-auto rounded-t-lg border-b object-contain')
                                    with ui.card_actions().props("align=right"):
                                        ui.button(icon="download", on_click=lambda p=str(plot_path): ui.download(p))\
                                            .props("flat dense size=sm color=primary round").tooltip("Download Plot")
                            else:
                                ui.label(f"Plot file not found: {plot_path.name}").classes('text-red-400 text-xs p-1')
                        except TypeError as te: 
                            logging.error(f"TypeError when creating Path for plot: {plot_path_str} - {te}", exc_info=True)
                            ui.label(f"Invalid plot path: {str(plot_path_str)[:50]}...").classes('text-orange-500 text-xs p-1')
                        except Exception as e_plot:
                            logging.error(f"Error rendering plot {plot_path_str}: {e_plot}", exc_info=True)
                            ui.label(f"Error for plot {str(plot_path_str)[:50]}...").classes('text-red-500 text-xs p-1')
            
            # --- FLATTEN AND DISPLAY FILES (similar logic) ---
            files_flattened = []
            if files_raw: # Check if files_raw is not empty
                for item in files_raw:
                    if isinstance(item, list):
                        for sub_item in item:
                            if isinstance(sub_item, (str, Path)):
                                files_flattened.append(sub_item)
                            else:
                                logging.warning(f"Skipping invalid sub-item in file path list: {sub_item} (type: {type(sub_item)})")
                    elif isinstance(item, (str, Path)):
                        files_flattened.append(item)
                    else:
                        logging.warning(f"Skipping invalid item in file path list: {item} (type: {type(item)})")

            if files_flattened:
                ui.markdown("###### Files & Data").classes("mt-2 text-sm")
                for file_path_str in files_flattened:
                    # ... (your existing logic for displaying files, using file_path_str) ...
                    # Ensure you use Path(file_path_str) within a try-except block here too.
                    try:
                        file_path = Path(file_path_str)
                        if file_path.is_file():
                            with ui.card().tight().classes("my-2 w-full shadow-lg"):
                                with ui.card_section().classes("flex justify-between items-center p-2"):
                                    ui.label(file_path.name).classes("font-semibold text-sm")
                                    ui.button(icon="download", on_click=lambda p=str(file_path): ui.download(p))\
                                        .props("flat dense size=sm color=primary round").tooltip("Download File")
                                if file_path.suffix.lower() in ['.csv', '.tsv']:
                                    ui.separator()
                                    with ui.card_section().classes("p-0"):
                                        try:
                                            df_preview = pd.read_csv(file_path, sep=',' if file_path.suffix.lower() == '.csv' else '\t')
                                            ui.table.from_pandas(df_preview.head(3)).classes('h-[150px] overflow-auto w-full bordered text-xs').props('dense flat bordered separator=cell')
                                        except Exception as e_df:
                                            ui.label(f"Preview failed: {e_df}").classes('text-orange-400 text-xs p-2')
                                elif file_path.suffix.lower() == '.html':
                                    ui.separator()
                                    with ui.card_section().classes("p-1"):
                                        try:
                                            with open(file_path, 'r', encoding='utf-8') as f_html:
                                                html_content = f_html.read()
                                            ui.html(html_content).classes('max-h-96 h-[350px] overflow-auto border w-full')
                                        except Exception as e_html:
                                            ui.label(f"HTML display failed: {e_html}").classes('text-orange-400 text-xs p-2')
                        else:
                            ui.label(f"Data file not found: {file_path.name}").classes('text-red-400 text-xs p-1')
                    except TypeError as te:
                        logging.error(f"TypeError when creating Path for file: {file_path_str} - {te}", exc_info=True)
                        ui.label(f"Invalid file path: {str(file_path_str)[:50]}...").classes('text-orange-500 text-xs p-1')
                    except Exception as e_file:
                        logging.error(f"Error rendering file entry {file_path_str}: {e_file}", exc_info=True)
                        ui.label(f"Error for file {str(file_path_str)[:50]}...").classes('text-red-500 text-xs p-1')

            if not plots_flattened and not files_flattened and not (middle_steps_str and middle_steps_str.startswith("####")):
                ui.label("No specific plots, files, or detailed agent steps for this message.")\
                    .classes("m-2 text-gray-500")


    # def update_chat_display(self): # Modified
    #     if not self.chat_container: return
    #     self.chat_container.clear()
    #     with self.chat_container:
    #         for i, msg_data in enumerate(self.messages):
    #             role, is_user = msg_data.get("role", "assistant"), msg_data.get("role") == "user"
    #             name = self.user_id if is_user else "Agent"; avatar_char = name[0].upper() if name else ('U' if is_user else 'A')
    #             timestamp_str = msg_data.get("timestamp", ""); stamp_display = ""
    #             if timestamp_str:
    #                 try: stamp_display = pd.to_datetime(timestamp_str).strftime('%I:%M %p') 
    #                 except Exception: stamp_display = timestamp_str 
                
    #             chat_message_props = (f"text-color=black bg-color={'blue-1' if is_user else 'grey-2'} name-color={'indigo-8' if is_user else 'deep-purple-8'} stamp-color=grey-7")
                
    #             with ui.chat_message(name=name, sent=is_user, stamp=stamp_display, avatar=f'https://robohash.org/{avatar_char}?set=set3&bgset=bg1' if is_user else f'https://robohash.org/{avatar_char}?set=set5&bgset=bg2').props(chat_message_props).classes('w-full rounded-lg shadow-sm'):
    #                 original_content_value = msg_data.get("content", "") # Get the content as is
    #                 msg_type = msg_data.get("type", "text")

    #                 # ---- MODIFICATION START ----
    #                 # Ensure content is a string before passing to ui.markdown
    #                 final_content_for_markdown: str
    #                 if isinstance(original_content_value, list):
    #                     # If content is a list (e.g., from old chat history), join its elements.
    #                     # map(str, ...) ensures all elements are strings before joining.
    #                     final_content_for_markdown = "\n".join(map(str, original_content_value))
    #                 else:
    #                     # If content is not a list, convert to string (it might already be a string).
    #                     final_content_for_markdown = str(original_content_value)
    #                 # ---- MODIFICATION END ----

    #                 ui.markdown(final_content_for_markdown).classes('text-sm link-styling') # Line 738 in new traceback

    #                 if role == "assistant":
    #                     with ui.row().classes("items-center -ml-1 mt-1"): # Group buttons
    #                         if msg_type == "text_with_attachments" and (bool(msg_data.get("plots")) or bool(msg_data.get("files")) or (msg_data.get("middle_steps") and msg_data.get("middle_steps").startswith("####"))):
    #                             ui.button("View Plots & Tables", icon="table_chart", on_click=lambda bound_idx=i: self.show_details_for_message(bound_idx)).props('flat color=teal rounded size=sm').classes('text-xs px-2 py-0.5') # Changed icon and color
                            
    #                         if not msg_data.get('bookmarked'):
    #                             ui.button(icon="bookmark_add", on_click=lambda msg_idx=i: self.add_bookmark(msg_idx)).props("flat dense round color=amber-8 size=sm").tooltip("Bookmark this response")
    #                         else:
    #                             ui.icon("bookmark_added", color="amber-8 size-5").classes("ml-2 cursor-default").tooltip("Bookmarked") # size-5 for icon
                        
    #                     if msg_type == "text_with_candidates":  # THIS COULD BE YOUR LINE 637
    #             # EVERYTHING INSIDE THIS 'if' BLOCK MUST BE INDENTED FURTHER
    #                         for cand_idx, cand in enumerate(msg_data.get("candidates", [])):
    #                             with ui.expansion(f"Candidate {cand_idx+1}: {cand.get('option', 'Option')}", icon='ballot').classes('w-full my-1 text-xs shadow-sm rounded-md border'):
    #                                 ui.markdown(f"**Expl:** {cand.get('explanation', '')[:150]}...").classes("p-1")
    #                                 if self.chat_input_field: 
    #                                     ui.button("Use this", icon='check_circle_outline', 
    #                                             on_click=lambda c=cand, ci=self.chat_input_field: (
    #                                                 ci.set_value(f"Regarding candidate '{c.get('option','')}': {c.get('explanation','')}... Please proceed."), 
    #                                                 ci.run_method('focus'))
    #                                             ).props(f'flat dense size=xs key="refine_{i}_{cand_idx}"').classes("m-1")
            
    #         # The 'if' for next_steps should be at the same indentation level as 
    #         # 'if msg_type == "text_with_candidates":' IF IT'S A SEPARATE, INDEPENDENT CHECK,
    #         # OR it could be an 'elif' or nested if its logic depends on the previous one.
    #         # In my last version, it was an independent check for next_steps.
    #                     if msg_data.get("next_steps"): # THIS IS LIKELY YOUR LINE 638
    #                         # EVERYTHING INSIDE THIS 'if' BLOCK MUST BE INDENTED FURTHER
    #                         with ui.row().classes("mt-2 gap-1 flex-wrap items-center"): # mt-2 for spacing
    #                             ui.markdown("**Next:**").classes("self-center text-xs mr-1 text-gray-700")
    #                             for step_idx, step in enumerate(msg_data["next_steps"][:3]):
    #                                 if self.chat_input_field:
    #                                     ui.button(step, 
    #                                             on_click=lambda s=step, ci=self.chat_input_field: (ci.set_value(s), ci.run_method('focus'))) \
    #                                         .props(f'flat dense no-caps key="next_step_{i}_{step_idx}"') \
    #                                         .classes('text-sm bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-full px-3 py-1')
    #     def scroll_chat_to_bottom():
    #         if self.chat_container and self.chat_container.client.has_socket_connection:
    #             chat_id = self.chat_container.id
    #             # Use ui.run_javascript to execute a script that finds the element by ID and scrolls it
    #             js_command = f"var el = getElement({chat_id}); if (el) {{ el.scrollTop = el.scrollHeight; }}"
    #             ui.run_javascript(js_command)
    #             logging.debug(f"Attempted to scroll chat container (ID: {chat_id}) to bottom via ui.run_javascript.")
    #         elif self.chat_container:
    #             logging.debug("Chat container exists but no socket connection for scroll.")
    #         else:
    #             logging.warning("Chat container does not exist when trying to schedule scroll.")

    #     # Use a short timer to ensure the JS runs after client has processed UI updates
    #     # and new message elements are rendered, making scrollHeight accurate.
    #     ui.timer(0.1, scroll_chat_to_bottom, once=True) # 0.1 second delay



    def update_chat_display(self):
        if not self.chat_container:
            logging.warning("Chat container not available for update_chat_display.")
            return
        
        self.chat_container.clear()
        with self.chat_container:
            for i, msg_data in enumerate(self.messages):
                role, is_user = msg_data.get("role", "assistant"), msg_data.get("role") == "user"
                name = self.user_id if is_user else "Agent"
                avatar_char = name[0].upper() if name else ('U' if is_user else 'A')
                
                chat_message_props = (
                    f"text-color=black bg-color={'blue-1' if is_user else 'grey-2'} "
                    f"name-color={'indigo-8' if is_user else 'deep-purple-8'}"
                    # stamp-color prop on q-chat-message is for its own stamp, which we are bypassing.
                )
                
                # We will NOT use the 'stamp' prop of ui.chat_message.
                # We will add our own ui.html element for the timestamp.
                with ui.chat_message(
                    name=name, 
                    sent=is_user, 
                    avatar=f'https://robohash.org/{avatar_char}?set=set5&bgset=bg1' if is_user else f'https://robohash.org/{avatar_char}?set=set3&bgset=bg2'
                ).props(chat_message_props).classes('w-full rounded-lg shadow-sm'):
                    
                    # This column will hold the main content and then our custom stamp below it.
                    with ui.column().classes('w-full no-wrap pa-0 ma-0'): # Ensure no extra padding/margin from this column
                        original_content_value = msg_data.get("content", "")
                        msg_type = msg_data.get("type", "text")
                        final_content_for_markdown: str
                        if isinstance(original_content_value, list):
                            final_content_for_markdown = "\\n".join(map(str, original_content_value))
                        else:
                            final_content_for_markdown = str(original_content_value)
                        
                        ui.markdown(final_content_for_markdown).classes('text-sm link-styling')

                        # --- CUSTOM CLIENT-SIDE TIMESTAMP RENDERING ---
                        raw_timestamp_str = msg_data.get("timestamp")
                        utc_iso_for_js = None 

                        if raw_timestamp_str and isinstance(raw_timestamp_str, str) and raw_timestamp_str.strip():
                            # Python pre-processing to ensure string is explicitly UTC for JS
                            try:
                                has_timezone_info = 'Z' in raw_timestamp_str.upper() or \
                                                    ('+' in raw_timestamp_str[10:]) or \
                                                    ('-' in raw_timestamp_str[10:] and raw_timestamp_str.count(':') >= 2)
                                if has_timezone_info:
                                    dt_obj = pd.to_datetime(raw_timestamp_str)
                                    utc_iso_for_js = dt_obj.tz_convert('UTC').isoformat()
                                else:
                                    utc_iso_for_js = raw_timestamp_str + "Z"
                            except Exception as e_parse:
                                logging.warning(f"PYTHON_DEBUG: Could not fully normalize timestamp '{raw_timestamp_str}', using as-is or Z-appended for JS: {e_parse}")
                                if 'T' in raw_timestamp_str and not ('Z' in raw_timestamp_str.upper() or '+' in raw_timestamp_str[10:] or ('-' in raw_timestamp_str[10:] and raw_timestamp_str.count(':') >=2 )):
                                    utc_iso_for_js = raw_timestamp_str + "Z"
                                else:
                                    utc_iso_for_js = raw_timestamp_str # Pass potentially problematic string if normalization failed

                        if utc_iso_for_js:
                            timestamp_dom_id = f"custom_ts_element_{uuid.uuid4().hex[:8]}"
                            
                            # This HTML element will contain our timestamp.
                            # We apply classes to make it look like a Quasar stamp.
                            # It's crucial this is styled and positioned correctly.
                            # It is now a sibling to the ui.markdown output, inside the ui.column.
                            ui.html(f'<div id="{timestamp_dom_id}" class="custom-timestamp-style w-full text-right"></div>')
                            
                            js_code_to_format_stamp = f"""
                                (function() {{
                                    var el = document.getElementById('{timestamp_dom_id}');
                                    var utcTimestampStr = '{utc_iso_for_js}';
                                    // console.log('[JS_DEBUG] Processing ID: {timestamp_dom_id}, UTC String: "' + utcTimestampStr + '"');
                                    if (el) {{
                                        try {{
                                            var date = new Date(utcTimestampStr);
                                            if (isNaN(date.getTime())) {{
                                                el.textContent = ''; // Or '[time N/A]'
                                                // console.warn('[JS_DEBUG] Invalid Date for ID {timestamp_dom_id} from:', utcTimestampStr);
                                            }} else {{
                                                el.textContent = date.toLocaleTimeString(undefined, {{ 
                                                    hour: 'numeric', 
                                                    minute: '2-digit', 
                                                    hour12: true 
                                                }});
                                            }}
                                        }} catch (e) {{
                                            el.textContent = ''; // Or '[time error]'
                                            // console.error('[JS_DEBUG] JS Exception for ID {timestamp_dom_id}:', e);
                                        }}
                                    }} else {{
                                        // console.warn('[JS_DEBUG] Timestamp DOM element NOT FOUND by ID:', '{timestamp_dom_id}');
                                    }}
                                }})();
                            """
                            ui.timer(0.15, lambda code=js_code_to_format_stamp: ui.run_javascript(code), once=True)
                        # --- END CUSTOM CLIENT-SIDE TIMESTAMP ---
                    
                    # ... (Your existing logic for assistant message buttons, candidates, next_steps) ...
                    # (Ensure this logic is placed correctly relative to the ui.column holding markdown and stamp)
                    if role == "assistant":
                        with ui.row().classes("items-center -ml-1 mt-1"): # This row is a sibling to the ui.column above
                            if msg_type == "text_with_attachments" and \
                               (bool(msg_data.get("plots")) or \
                                bool(msg_data.get("files")) or \
                                (msg_data.get("middle_steps") and msg_data.get("middle_steps").startswith("####"))):
                                ui.button("View Plots & Tables", icon="table_chart", on_click=lambda bound_idx=i: self.show_details_for_message(bound_idx))\
                                  .props('flat color=teal rounded size=sm').classes('text-xs px-2 py-0.5')
                            if not msg_data.get('bookmarked'):
                                ui.button(icon="bookmark_add", on_click=lambda msg_idx=i: self.add_bookmark(msg_idx))\
                                  .props("flat dense round color=amber-8 size=sm").tooltip("Bookmark this response")
                            else:
                                ui.icon("bookmark_added", color="amber-8 size-5").classes("ml-2 cursor-default").tooltip("Bookmarked")
                        if msg_type == "text_with_candidates":
                             for cand_idx, cand in enumerate(msg_data.get("candidates", [])):
                                with ui.expansion(f"Candidate {cand_idx+1}: {cand.get('option', 'Option')}", icon='ballot').classes('w-full my-1 text-xs shadow-sm rounded-md border'):
                                    ui.markdown(f"**Expl:** {cand.get('explanation', '')[:150]}...").classes("p-1")
                                    if self.chat_input_field:
                                        ui.button("Use this", icon='check_circle_outline', 
                                            on_click=lambda c=cand, ci=self.chat_input_field: (
                                                ci.set_value(f"Regarding candidate '{c.get('option','')}': {c.get('explanation','')}... Please proceed."), 
                                                ci.run_method('focus'))
                                            ).props(f'flat dense size=xs key="refine_{i}_{cand_idx}"').classes("m-1")
                        if msg_data.get("next_steps"):
                            with ui.row().classes("mt-2 gap-1 flex-wrap items-center"):
                                ui.markdown("**Next:**").classes("self-center text-xs mr-1 text-gray-700")
                                for step_idx, step in enumerate(msg_data["next_steps"][:3]):
                                    if self.chat_input_field:
                                        ui.button(step, 
                                            on_click=lambda s=step, ci=self.chat_input_field: (ci.set_value(s), ci.run_method('focus'))) \
                                            .props(f'flat dense no-caps key="next_step_{i}_{step_idx}"') \
                                            .classes('text-sm bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-full px-3 py-1')


        # Scroll to bottom logic
        def scroll_chat_to_bottom():
            if self.chat_container and self.chat_container.client.has_socket_connection:
                chat_id = self.chat_container.id
                js_command = f"var el = getElement({chat_id}); if (el) {{ el.scrollTop = el.scrollHeight; }}"
                ui.run_javascript(js_command)
        ui.timer(0.1, scroll_chat_to_bottom, once=True)

    # Place this method inside your NiceGuiApp class
    def _handle_drawer_escape_key(self, e): # 'e' is the event argument from ui.keyboard
        """Handles keyboard events to close the left drawer on Escape key press (on keydown)."""
        try:
            key_obj = getattr(e, 'key', None)
            action_obj = getattr(e, 'action', None)

            is_escape = False
            if key_obj:
                # Prioritize boolean flag if available (e.g_ e.key.escape)
                if hasattr(key_obj, 'escape') and key_obj.escape is True:
                    is_escape = True
                # Fallback to checking the key name if the boolean flag isn't present/true
                elif hasattr(key_obj, 'name') and isinstance(key_obj.name, str) and key_obj.name.lower() == 'escape':
                    is_escape = True
            
            is_keydown = False
            if action_obj and hasattr(action_obj, 'keydown') and action_obj.keydown is True:
                is_keydown = True
            
            if is_escape and is_keydown: # Process only on keydown of Escape
                if self.left_drawer and self.left_drawer.value: # Check if drawer exists and is open
                    self.left_drawer.value = False
                    # import logging # Make sure logging is imported in your class/module
                    # logging.debug("Left drawer closed via Escape key.") # Optional logging
        except AttributeError:
            # Gracefully handle cases where 'e' or its attributes might not have the expected structure
            # import logging
            # logging.debug(f"Key event with unexpected structure for Esc handling: {e}")
            pass
    
    # def build_ui(self):
    #     ui.add_head_html("""
    #         <style>
    #             .link-styling a { color: #3f51b5; text-decoration: underline; } .link-styling a:hover { color: #283593; }
    #             .dense-table .q-table th, .dense-table .q-table td { padding: 4px 8px !important; }
    #             .middle-steps-content { font-size: 0.75rem; max-height: 500px; overflow-y: auto; }
    #             .middle-steps-content pre { white-space: pre-wrap !important; word-break: break-all; overflow-x: auto; background-color: #f0f4f8; padding: 0.5rem; border-radius: 0.25rem; border: 1px solid #e2e8f0; font-family: monospace; }
    #             .middle-steps-content code { font-family: monospace; }
    #         </style>
    #     """)
    #     self.left_drawer = ui.left_drawer(elevated=True, top_corner=True, bottom_corner=True).props('overlay breakpoint=lg').style('background-color: #f4f6f8;').classes('p-4 w-80 lg:w-96 border-r')
    #     with ui.header(elevated=True).style('background-color: #303f9f;').classes('items-center justify-between text-white q-px-md'):
    #         ui.label("Galaxy Chat Analysis").classes("text-xl md:text-2xl font-semibold tracking-wide")
    #         if self.left_drawer: ui.button(icon='menu', on_click=self.left_drawer.toggle).props('flat round color=white')

    #     with self.left_drawer:
    #         ui.label("Configuration").classes("text-lg font-semibold mb-3 text-indigo-800")
    #         self.sidebar_api_status_label = ui.label("Agent: Unknown").classes("mb-3 text-xs p-1 rounded")
    #         self.model_select_element = ui.select(self.MODEL_OPTIONS_SELECT, label="LLM Model", value=self.selected_model_id, on_change=self.handle_model_change).props("outlined dense emit-value map-options").classes("w-full mb-3")
    #         with ui.expansion("API Keys", icon="key", value=False).classes("w-full mb-3 text-sm"): # ... API Key inputs ...
    #             self.openai_key_input = ui.input(label="OpenAI API Key", password=True, value=self.openai_api_key, on_change=lambda e: setattr(self, 'openai_api_key', e.value)).props("dense outlined clearable")
    #             ui.button("Save OpenAI", on_click=self.save_openai_key, icon="save").classes("w-full mt-1").props("color=indigo-6 dense size=sm")
    #             self.groq_key_input = ui.input(label="Groq API Key", password=True, value=self.groq_api_key, on_change=lambda e: setattr(self, 'groq_api_key', e.value)).props("dense outlined clearable mt-2")
    #             ui.button("Save Groq", on_click=self.save_groq_key, icon="save").classes("w-full mt-1").props("color=indigo-6 dense size=sm")

    #         ui.separator().classes("my-3")
    #         ui.label("Dataset").classes("text-md font-semibold mb-2 text-indigo-700") # ... Dataset upload ...
    #         ui.upload(label="Upload New Dataset", auto_upload=False, on_upload=self.handle_upload, max_file_size=200 * 1024 * 1024).props("accept=.csv,.tsv,.h5ad,.xlsx,.xls,.json,.parquet,.h5,.fa,.fasta,.vcf,.gtf,.gff,.bed").classes("w-full mb-3")

    #         ui.label("Analysis Actions").classes("text-md font-semibold mt-3 mb-2 text-indigo-700") # ... EDA button ...
    #         ui.button("Run Full EDA", on_click=self.run_eda_action, icon="query_stats").classes("w-full mb-1").props("color=deep-purple-6 dense")
    #         ui.separator().classes("my-3")

    #         # Bookmarks Section
    #         with ui.expansion("⭐ Bookmarks", icon="bookmarks", value=True).classes("w-full text-sm"):
    #             self.bookmarks_container = ui.column().classes("w-full max-h-100 overflow-y-auto gap-1") # Scrollable bookmark list
    #             self.update_sidebar_bookmarks() # Initial population

    #     with ui.splitter(value=60, reverse=False, limits=(30,70)).classes('w-full h-[calc(100vh-110px)] no-wrap') as main_splitter:
    #         with main_splitter.before:
    #             with ui.column().classes("w-full h-full p-0 items-stretch no-wrap items-stretch overflow-hidden"):
    #                 self.chat_container = ui.column().classes("w-full flex-grow overflow-y-auto p-2 md:p-3 bg-gray-100")
    #                 with ui.row().classes("w-full p-2 bg-slate-200 items-center border-t"):
    #                     self.chat_input_field = ui.input(placeholder="Ask about the dataset...").props("bg-color=white outlined dense clearable rounded").classes("flex-grow").on('keydown.enter', lambda: self.handle_user_input(self.chat_input_field.value), throttle=0.5)
    #                     ui.button(icon="send", on_click=lambda: self.handle_user_input(self.chat_input_field.value)).props("round color=indigo-6 dense unelevated")
    #                     ui.label("This agent can make mistakes, so double-check it.") \
    #                         .classes("w-full text-xs text-gray-600 p-2 text-center bg-slate-100 border-t flex-shrink-0")
    #         with main_splitter.after:
    #             with ui.column().classes("w-full h-full items-stretch overflow-y-auto bg-slate-50"): # This makes the whole right pane scroll
    #                 ui.label("Details & Preview").classes("text-md font-semibold mb-2 text-gray-700 sticky top-0 bg-slate-100/95 backdrop-blur-sm z-10 p-3 border-b shadow-sm")
    #                 with ui.column().classes("p-2 md:p-3 flex-grow"): # Content area below sticky header
                        
    #                     self.details_container = ui.column().classes("w-full flex-grow p-2 border rounded-lg bg-white shadow mt-2 min-h-[200px]")
    #                     self.dataset_preview_area = ui.column().classes("w-full mb-3 p-2 border rounded-lg bg-white shadow")
        
    #     app.on_connect(self.on_page_load_actions)
    #     app.on_disconnect(self.on_page_unload_actions)



    def build_ui(self):
        ui.add_head_html("""
            <style>
                .link-styling a { color: #3f51b5; text-decoration: underline; } .link-styling a:hover { color: #283593; }
                .dense-table .q-table th, .dense-table .q-table td { padding: 4px 8px !important; }
                .middle-steps-content { font-size: 0.75rem; max-height: 500px; overflow-y: auto; }
                .middle-steps-content pre { white-space: pre-wrap !important; word-break: break-all; overflow-x: auto; background-color: #f0f4f8; padding: 0.5rem; border-radius: 0.25rem; border: 1px solid #e2e8f0; font-family: monospace; }
                .middle-steps-content code { font-family: monospace; }
                .custom-timestamp-style {
                    font-size: 0.75em; /* Typical stamp size */
                    color: #757575;    /* Quasar's grey-7, a common stamp color */
                    margin-top: 4px;   /* Space from the message content above */
                    line-height: 1;    /* Adjust as needed */
                    width: 100%;
                    text-align: right; /* Common for stamps */
                    padding-right: 8px; /* Align with typical bubble padding */
                }
            </style>
        """)
        self.left_drawer = ui.left_drawer(elevated=True, top_corner=True, bottom_corner=True)\
            .props('overlay breakpoint=lg').style('background-color: #f4f6f8;')\
            .classes('p-4 w-80 lg:w-96 border-r')
        
        # Header with menu button on the right (as per your provided code)
        with ui.header(elevated=True).style('background-color: #303f9f;').classes('items-center text-white q-px-md'):
            if self.left_drawer: 
                ui.button(icon='menu', on_click=self.left_drawer.toggle).props('flat round color=white')

            ui.label("Galaxy Chat Analysis").classes("text-xl md:text-2xl font-semibold tracking-wide")
            
        with self.left_drawer:
            # ---- MODIFIED: Added Row for Title and Close Button ----
            with ui.row().classes("w-full items-center justify-between no-wrap mb-2"): # Use mb-2 here
                ui.label("Configuration").classes("text-lg font-semibold text-indigo-800") # Removed mb-3
                ui.button(icon='close', on_click=lambda: setattr(self.left_drawer, 'value', False)) \
                    .props('flat round dense color=grey-7').tooltip("Close Sidebar")
            # ---- END MODIFICATION ----
            
            self.sidebar_api_status_label = ui.label("Agent: Unknown").classes("mb-3 text-xs p-1 rounded")
            self.model_select_element = ui.select(self.MODEL_OPTIONS_SELECT, label="LLM Model", value=self.selected_model_id, on_change=self.handle_model_change).props("outlined dense emit-value map-options").classes("w-full mb-3")
            with ui.expansion("API Keys", icon="key", value=False).classes("w-full mb-3 text-sm"):
                self.openai_key_input = ui.input(label="OpenAI API Key", password=True, value=self.openai_api_key, on_change=lambda e: setattr(self, 'openai_api_key', e.value)).props("dense outlined clearable")
                ui.button("Save OpenAI", on_click=self.save_openai_key, icon="save").classes("w-full mt-1").props("color=indigo-6 dense size=sm")
                self.groq_key_input = ui.input(label="Groq API Key", password=True, value=self.groq_api_key, on_change=lambda e: setattr(self, 'groq_api_key', e.value)).props("dense outlined clearable mt-2")
                ui.button("Save Groq", on_click=self.save_groq_key, icon="save").classes("w-full mt-1").props("color=indigo-6 dense size=sm")

            ui.separator().classes("my-3")
            ui.label("Dataset").classes("text-md font-semibold mb-2 text-indigo-700")
            ui.upload(label="Upload New Dataset", auto_upload=False, on_upload=self.handle_upload, max_file_size=200 * 1024 * 1024).props("accept=.csv,.tsv,.h5ad,.xlsx,.xls,.json,.parquet,.h5,.fa,.fasta,.vcf,.gtf,.gff,.bed").classes("w-full mb-3")

            ui.label("Analysis Actions").classes("text-md font-semibold mt-3 mb-2 text-indigo-700")
            ui.button("Run Full EDA", on_click=self.run_eda_action, icon="query_stats").classes("w-full mb-1").props("color=deep-purple-6 dense")
            ui.separator().classes("my-3")

            with ui.expansion("⭐ Bookmarks", icon="bookmarks", value=True).classes("w-full text-sm"):
                # ---- MODIFIED: max-h-100 to max-h-96 (standard Tailwind class) ----
                self.bookmarks_container = ui.column().classes("w-full max-h-96 overflow-y-auto gap-1") 
                self.update_sidebar_bookmarks()

        with ui.splitter(value=60, reverse=False, limits=(30,70)).classes('w-full h-[calc(100vh-110px)] no-wrap') as main_splitter:
            with main_splitter.before:
                # ---- MODIFIED: Corrected classes for flex layout ----
                # Replaced repeated 'items-stretch' with 'flex flex-col' and added 'min-h-0'
                with ui.column().classes("w-full h-full p-0 flex flex-col no-wrap items-stretch overflow-hidden min-h-0"):
                    # ---- MODIFIED: Added min-h-0 to chat_container ----
                    self.chat_container = ui.column().classes("w-full flex-grow overflow-y-auto p-2 md:p-3 bg-gray-100 min-h-0")
                    
                    # Chat input row - added flex-shrink-0 for stability
                    with ui.row().classes("w-full p-2 bg-slate-200 items-center border-t flex-shrink-0"):
                        self.chat_input_field = ui.input(placeholder="Ask about the dataset...")\
                            .props("bg-color=white outlined dense clearable rounded").classes("flex-grow")\
                            .on('keydown.enter', lambda: self.handle_user_input(self.chat_input_field.value), throttle=0.5)
                        ui.button(icon="send", on_click=lambda: self.handle_user_input(self.chat_input_field.value))\
                            .props("round color=indigo-6 dense unelevated")
                    
                    # ---- MODIFIED: Disclaimer moved out of the input row to be below it ----
                    ui.label("This agent can make mistakes, so double-check it.") \
                        .classes("w-full text-xs text-gray-600 p-2 text-center bg-slate-100 border-t flex-shrink-0")
            
            with main_splitter.after:
                # This structure makes the whole right pane scroll, as per your provided version.
                # The "Details & Preview" label will scroll with it.
                with ui.column().classes("w-full h-full items-stretch overflow-y-auto bg-slate-50"): 
                    ui.label("Details & Preview").classes("text-md font-semibold mb-2 text-gray-700 sticky top-0 bg-slate-100/95 backdrop-blur-sm z-10 p-3 border-b shadow-sm")
                    with ui.column().classes("p-2 md:p-3 flex-grow"): 
                        # Order: Details then Dataset Preview (as per earlier request)
                        # Keeping your margins, but if details_container is first, mt-2 might not be needed.
                        self.details_container = ui.column().classes("w-full flex-grow p-2 border rounded-lg bg-white shadow mt-2 min-h-[200px]")
                        self.dataset_preview_area = ui.column().classes("w-full mb-3 p-2 border rounded-lg bg-white shadow") 
        
        app.on_connect(self.on_page_load_actions)
        app.on_disconnect(self.on_page_unload_actions)

    async def on_page_load_actions(self, client: Client):
        logging.info(f"Client connected (User: {self.user_id}). Loading initial actions.")
        dataset_loaded = False
        if self.initial_dataset_path_from_arg and self.initial_dataset_path_from_arg.exists():
            self.current_dataset_file_path = self.initial_dataset_path_from_arg; self.current_dataset_display_name = self.current_dataset_file_path.name
            self.current_input_data_type = self.cli_args.input_data_type
            ui.notify(f"Loading dataset from arg: {self.current_dataset_display_name}", type='info', timeout=2000)
            await self.preview_loaded_or_uploaded_dataset(); dataset_loaded = True
        elif self.current_dataset_file_path and self.current_dataset_file_path.exists():
            ui.notify(f"Restoring session with: {self.current_dataset_display_name}", type='info', timeout=2000)
            await self.preview_loaded_or_uploaded_dataset(); dataset_loaded = True
        if not dataset_loaded and self.dataset_preview_area:
            self.dataset_preview_area.clear();
            with self.dataset_preview_area: ui.label("Upload dataset or provide via CLI to start.").classes("text-gray-500 m-2")
        self.update_chat_display(); self.update_details_pane(); self.update_sidebar_bookmarks(); self.try_initialize_agent()

    def on_page_unload_actions(self, client: Client):
        logging.info(f"Client disconnected (User: {self.user_id}). Saving history.")
        self.save_chat_history()

    def handle_model_change(self, e):
        self.selected_model_id = e.value
        self.selected_model_name = self.MODEL_OPTIONS_SELECT.get(self.selected_model_id, self.selected_model_id)
        ui.notify(f"Model set to: {self.selected_model_name}", type='info', position='top-right', timeout=2000)
        self.try_initialize_agent()

    def save_openai_key(self):
        if self.openai_key_input:
            self.openai_api_key = self.openai_key_input.value or ""
            save_key_to_specific_file(OPENAI_API_KEY_FILE, self.openai_api_key)
            ui.notify("OpenAI Key " + ("saved." if self.openai_api_key else "cleared."), type='positive' if self.openai_api_key else 'info')
            self.try_initialize_agent()

    def save_groq_key(self):
        if self.groq_key_input:
            self.groq_api_key = self.groq_key_input.value or ""
            save_key_to_specific_file(GROQ_API_KEY_FILE, self.groq_api_key)
            ui.notify("Groq Key " + ("saved." if self.groq_api_key else "cleared."), type='positive' if self.groq_api_key else 'info')
            self.try_initialize_agent()

    async def run_eda_action(self):
        if not self.agent or not self.current_dataset_file_path:
            ui.notify("Agent or dataset not ready for EDA.", type='warning'); return
        eda_user_query = ("Perform a comprehensive EDA: summary statistics, missing values, data types, "
                          "correlation matrix, distributions for numerical, counts for categorical. Conclude with 3-5 insights.")
        ui.notify("Starting EDA...", type='info')
        await self.handle_user_input(eda_user_query)


# --- CLI Argument Parsing & App Run ---
if __name__ in {"__main__", "__mp_main__"}:
    parser = argparse.ArgumentParser(description="Galaxy Chat Analysis with NiceGUI")
    parser.add_argument("--user_id", nargs='?', default=f"user_{uuid.uuid4().hex[:6]}", help="User ID (defaults to a random ID).")
    parser.add_argument("--openai_key_file", dest="cli_openai_key_file_path", help="Path to OpenAI API key file.")
    parser.add_argument("--groq_key_file", dest="cli_groq_key_file_path", help="Path to Groq API key file.")
    parser.add_argument("--chat_history", dest="chat_history_path", default=str(DEFAULT_CHAT_HISTORY_FILE), help="Path to chat history JSON file.")
    parser.add_argument("--output_dir", dest="generate_file_path", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated files (plots, data).")
    parser.add_argument("--input_file", dest="input_file_path", help="Path to an initial dataset file to load.")
    parser.add_argument("--input_type", dest="input_data_type", default="csv", help="Type of the initial dataset file (e.g., csv, tsv, h5ad).")
    cli_args = parser.parse_args()

    app_instance = NiceGuiApp(user_id=cli_args.user_id, cli_args_ns=cli_args)

    @ui.page('/')
    def main_page_entry(client: Client): 
        app_instance.build_ui()

    ui.run(title="Galaxy Chat Analysis", storage_secret=str(uuid.uuid4()),
           port=8090, reload=os.environ.get('NICEGUI_RELOAD', 'true').lower() != 'false',
           uvicorn_logging_level='info', dark=False,
           favicon=SCRIPT_PATH / "favicon.ico") # Changed logging to info for more Uvicorn details if needed during dev