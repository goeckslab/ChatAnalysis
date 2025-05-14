import os
import re
import base64
import pandas as pd
from PIL import Image
from collections import deque
from dotenv import load_dotenv
import streamlit as st
from smolagents import CodeAgent, LiteLLMModel
import json
import uuid
import logging
import sys
from pathlib import Path
import psycopg2

# Set logging level to DEBUG for detailed logs
# logging.basicConfig(level=logging.DEBUG)

OPENAI_API_KEY_FILE = "user_config_openai.key"
GROQ_API_KEY_FILE = "user_config_groq.key"

load_dotenv()

st.set_page_config(
    page_title="Galaxy Chat Analysis",
    page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'favicon.ico'),
    layout="wide",)

st.markdown("""
<style>
    /* Target the specific section you identified (e.g., the second one) */
    div[data-testid="stAppViewContainer"] > section:nth-of-type(2) {
        max-width: 60% !important;   /* << ADJUST THIS VALUE to your desired width */
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 2.5rem;           /* Optional: Adjust side padding */
        padding-right: 2.5rem;          /* Optional: Adjust side padding */
        /* You can add a light border here too if you want to see its final bounds, e.g.: */
        /* border: 1px solid lightgrey !important; */
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* General style for all navigation links to ensure consistency */
    [data-testid="stSidebarNav"] ul li a {
        display: block; /* Makes the entire area clickable and styleable */
        padding: 0.5rem 0.75rem; /* Adjust padding: top/bottom left/right */
        margin-bottom: 0.2rem; /* Space between items */
        border-radius: 0.375rem; /* Rounded corners */
        transition: background-color 0.2s ease-in-out, color 0.2s ease-in-out, border-left 0.2s ease-in-out;
        color: #4A5568; /* Default text color for non-active items (a grayish tone) */
        text-decoration: none; /* Remove underline from links */
        border-left: 4px solid transparent; /* Placeholder for active border */
    }

    /* Style for the CURRENTLY ACTIVE navigation link */
    [data-testid="stSidebarNav"] ul li a[aria-current="page"] {
        background-color: #4299E1 !important; /* A brighter, more prominent blue */
        color: white !important;             /* White text for contrast */
        font-weight: 700 !important;         /* Even bolder */
        border-left: 4px solid #2B6CB0 !important; /* Prominent left border (darker blue) */
    }

    /* Hover effect for NON-ACTIVE links */
    [data-testid="stSidebarNav"] ul li a:not([aria-current="page"]):hover {
        background-color: #E2E8F0; /* Light gray background on hover */
        color: #2B6CB0 !important; /* Darker blue text on hover */
        border-left: 4px solid #A0AEC0; /* Subtle border on hover for non-active */
    }

    /* Optional: Slightly dim non-active links if you want extreme focus on active */
    /*
    [data-testid="stSidebarNav"] ul li a:not([aria-current="page"]) {
        opacity: 0.75;
    }
    [data-testid="stSidebarNav"] ul li a:not([aria-current="page"]):hover {
        opacity: 1;
    }
    */
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .stButton > button {
        width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def create_agent(api_key, model_id):
    model = LiteLLMModel(model_id=model_id, api_key=api_key)
    return CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=[
            "pandas", "numpy", "matplotlib", "seaborn", "scipy",
            "sklearn", "pycaret", "plotly", "joblib", "io", "xgboost",
            "lightgbm", "catboost",
        ],
        max_steps=20,
    )

# Place these functions globally

def save_key_to_specific_file(file_path: str, key_value: str):
    """Saves a key to a specific text file. Overwrites existing file."""
    try:
        # Ensure the directory exists if file_path includes directories
        dir_name = os.path.dirname(file_path)
        if dir_name: # If there's a directory part
            os.makedirs(dir_name, exist_ok=True)
            
        with open(file_path, "w") as f:
            f.write(key_value)
        logging.info(f"API key saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving API key to {file_path}: {e}")

def load_key_from_specific_file(file_path: str) -> str | None:
    """Loads an API key from a specific text file. Returns None if error or not found."""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                key = f.read().strip()
                if key:  # Ensure the key is not just whitespace
                    logging.info(f"API key loaded from {file_path}")
                    return key
                else:
                    logging.info(f"{file_path} exists but is empty or contains only whitespace.")
                    # Optionally, you could delete an empty key file here if desired:
                    # os.remove(file_path)
                    # logging.info(f"Removed empty key file: {file_path}")
        else:
            logging.info(f"Local API key file not found: {file_path}")
    except Exception as e:
        logging.error(f"Error loading API key from {file_path}: {e}")
    return None

def fix_code_block_formatting(text):
    """
    Inserts a newline after any occurrence of triple backticks (```)
    that is not immediately followed by a newline.
    """
    # The negative lookahead (?!\n) makes sure we only match when there isn't a newline
    fixed_text = re.sub(r"```(?!\n)", "```\n", text)
    return fixed_text

def split_text_preserving_order(text):
    """
    Splits text into a list of segments preserving the original order.
    Each segment is a tuple (segment_type, segment_text),
    where segment_type is either 'text' or 'code'.
    Code blocks are assumed to be wrapped in triple backticks.
    """
    pattern = r"```(?:\w+)?\n(.*?)\n```"
    segments = []
    last_index = 0
    for match in re.finditer(pattern, text, flags=re.DOTALL):
        start, end = match.span()
        # Capture any text before this code block.
        if start > last_index:
            text_segment = text[last_index:start].strip()
            if text_segment:
                segments.append(("text", text_segment))
        # Capture the code block.
        code_segment = match.group(1)
        segments.append(("code", code_segment))
        last_index = end
    # Capture any remaining text after the last code block.
    if last_index < len(text):
        remaining_text = text[last_index:].strip()
        if remaining_text:
            segments.append(("text", remaining_text))
    return segments


def clean_text(text):
    """Fix LLM responses where words are split into single-character lines."""
    lines = text.splitlines()
    cleaned_lines = []
    
    temp_word = ""
    for line in lines:
        words = line.strip().split()
        
        # If the line contains only a single character, merge it
        if len(words) == 1 and len(words[0]) == 1:
            temp_word += words[0]
        else:
            # If we have an accumulated word, add it before this line
            if temp_word:
                cleaned_lines.append(temp_word)
                temp_word = ""
            cleaned_lines.append(line)

    # Append any remaining word
    if temp_word:
        cleaned_lines.append(temp_word)

    return "\n".join(cleaned_lines)

def check_db_env_vars():
    """
    Confirm that all required PostgreSQL environment variables are set.
    """
    required_vars = ["PG_HOST_DA", "PG_DB_DA", "PG_USER_DA", "PG_PASSWORD_DA"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logging.warning("Missing required DB environment variables: %s", missing_vars)
        return False
    return True

def check_db_connection():
    """
    Check if the environment variables are set and if a connection to the DB can be established.
    Return True if successful, False otherwise.
    """
    if not check_db_env_vars():
        return False
    try:
        conn = get_db_connection()
        conn.close()
        return True
    except Exception as e:
        logging.error("Database connection failed: %s", e)
        return False


def get_db_connection():
    """
    Establish a connection to the PostgreSQL database using parameters from environment variables.
    If the environment variables are not set, default values are used.
    """
    if check_db_env_vars():
        
        conn = psycopg2.connect(
            host=os.environ["PG_HOST_DA"],
            database=os.environ["PG_DB_DA"],
            user=os.environ["PG_USER_DA"],
            password=os.environ["PG_PASSWORD_DA"]
        )
        return conn

def init_feedback_db():
    """
    Create the message_feedback table in PostgreSQL if it does not already exist.
    The table now includes a dataset_path column to record the input dataset path.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS message_feedback (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            feedback TEXT NOT NULL,
            comment TEXT,
            dataset_path TEXT,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def store_message_feedback(user_id, question, answer, feedback, dataset_path, comment=""):
    """
    Insert a new feedback record into the PostgreSQL database, including the dataset path.
    Returns the generated feedback record ID.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO message_feedback (user_id, question, answer, feedback, comment, dataset_path)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;
        """,
        (user_id, question, answer, feedback, comment, dataset_path)
    )
    feedback_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return feedback_id

def update_feedback_comment(feedback_id, comment):
    """
    Update the comment field for an existing feedback record.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE message_feedback SET comment = %s WHERE id = %s;",
        (comment, feedback_id)
    )
    conn.commit()
    cur.close()
    conn.close()


class StreamlitApp:
    def __init__(self, 
                 agent, 
                 user_id, 
                 output_dir="outputs_smolagents", 
                 dataset_file_path=None,
                 chat_history_file="chat_history.json",
                 input_data_type="csv"):
        self.user_id = user_id
        self.agent = agent
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if "memory" not in st.session_state:
            st.session_state["memory"] = deque(maxlen=30)
        if "bookmarks" not in st.session_state:
            st.session_state["bookmarks"] = []
        self.dataset_file = None
        self.dataset_file_path = dataset_file_path
        self.chat_hisory_file = chat_history_file
        self.input_data_type = input_data_type

        self.current_data_object = None # Will store the loaded pandas DataFrame, AnnData, etc.
        self.summary_stats_csv_path = None

    def load_dataset(self, file):
        path = Path(file)
        input_data_type = st.session_state.get("input_data_type", "csv")
        
        if path.is_file() and input_data_type == 'csv':
            self.dataset_file = True
            return pd.read_csv(file)

        elif input_data_type == "tsv":
            self.dataset_file = True
            df = pd.read_csv(file, sep="\t")
            return df

        elif input_data_type == "h5ad":
            import anndata
            self.dataset_file = True
            return anndata.read_h5ad(file)

        elif input_data_type in ("xlsx", "xls"):
            self.dataset_file = True
            return pd.read_excel(file)

        elif input_data_type == "json":
            self.dataset_file = True
            return pd.read_json(file)

        elif input_data_type == "parquet":
            self.dataset_file = True
            return pd.read_parquet(file)

        elif input_data_type == "h5":
            self.dataset_file = True
            return pd.read_hdf(file)

        elif input_data_type in ("fa", "fasta"):
            from Bio import SeqIO
            self.dataset_file = True
            return list(SeqIO.parse(file, "fasta"))

        elif input_data_type == "vcf":
            import pysam
            self.dataset_file = True
            return pysam.VariantFile(file)

        elif input_data_type in ("gtf", "gff"):
            import gffutils
            db = gffutils.create_db(
                file,
                dbfn=":memory:",
                force=True,
                keep_order=True,
                merge_strategy="merge",
                sort_attribute_values=True
            )
            self.dataset_file = False
            return db

        elif input_data_type == "bed":
            self.dataset_file = True
            return pd.read_csv(file, sep="\t", header=None)

        else:
            raise ValueError("Unsupported file format. Please provide a supported data file.")


    def generate_and_save_pandas_summary_csv(self, data) -> str | None:
        if data is None or not isinstance(data, pd.DataFrame):
            logging.warning("Attempted to generate pandas summary, but current_data_object is not a DataFrame.")
            return None

        dataframe = data
        original_filename_for_summary = "dataset" # Default
        current_dataset_path = st.session_state.get("analysis_file_path")
        if current_dataset_path:
            original_filename_for_summary = os.path.splitext(os.path.basename(current_dataset_path))[0]

        try:
            summary_df = dataframe.describe(include='all')
            
            summary_filename = f"summary_stats_for_{original_filename_for_summary}_{uuid.uuid4().hex[:6]}.csv"
            os.makedirs(self.output_dir, exist_ok=True) # Ensure output dir exists
            summary_csv_path = os.path.join(self.output_dir, summary_filename)
            
            summary_df.to_csv(summary_csv_path, index=True) 
            logging.info(f"Pandas summary statistics saved to: {summary_csv_path}")
            return summary_csv_path
        except Exception as e:
            logging.error(f"Error generating/saving pandas summary CSV for {original_filename_for_summary}: {e}", exc_info=True)
            return None
        

    def preview_dataset(self, file):
        try:
            
            data = self.load_dataset(file)
            # For pandas DataFrame types (CSV, TSV, Excel, Parquet, HDF, BED)
            if isinstance(data, pd.DataFrame):
                st.markdown("Dataset Preview (First 5 Rows)")
                # AgGrid(data.head(5), height=220, enable_enterprise_modules=False)
                st.dataframe(data.head(5))
            
            # For AnnData objects (e.g., .h5ad files)
            elif hasattr(data, "obs") and hasattr(data, "var"):
                st.markdown("AnnData Observations Preview (First 5 Rows)")
                st.dataframe(data.obs.head())
                st.markdown("AnnData Variables Preview (First 5 Rows)")
                st.dataframe(data.var.head())
            
            # For FASTA files (list of sequences)
            elif isinstance(data, list):
                st.markdown("FASTA Sequences Preview")
                # Display a limited number of sequences, e.g., the first 5
                for i, record in enumerate(data[:5], start=1):
                    st.markdown(f"**Sequence {i}:**")
                    st.text(str(record.seq))
            
            # For VCF files (using pysam VariantFile)
            elif hasattr(data, "header"):
                st.markdown("VCF Header")
                st.text(str(data.header))
                # Optionally, iterate over the first few records:
                st.markdown("VCF Records Preview")
                for i, rec in enumerate(data.fetch(), start=1):
                    st.text(str(rec))
                    if i >= 5:
                        break
            
            # For GTF/GFF files using gffutils (in-memory DB)
            elif hasattr(data, "all_features"):
                st.markdown("GTF/GFF Features Preview")
                features = list(data.all_features())
                preview_features = features[:5] if len(features) >= 5 else features
                for i, feature in enumerate(preview_features, start=1):
                    st.markdown(f"**Feature {i}:** {feature}")
            
            else:
                st.warning("Preview not supported for this file type.")
            
            current_data_type = self.input_data_type
            pandas_compatible_types = ['csv', 'tsv', 'xlsx', 'xls', 'json', 'parquet', 'h5', 'bed']
            if current_data_type in pandas_compatible_types and isinstance(data, pd.DataFrame):
                if not st.session_state.get("summary_stats_csv_path", None):
                    generated_summary_path = self.generate_and_save_pandas_summary_csv(data)
                    st.session_state["summary_stats_csv_path"] = generated_summary_path
                else:
                    generated_summary_path = st.session_state.get("summary_stats_csv_path", None)
                
                if generated_summary_path:
                    self.summary_stats_csv_path = generated_summary_path # Store path
                    st.markdown("#### Summary Statistics")
                    try:
                        # Read with index_col=0 because df.describe() often has meaningful row labels (like 'count', 'mean')
                        summary_display_df = pd.read_csv(self.summary_stats_csv_path, index_col=0) 
                        st.dataframe(summary_display_df)
                        
                        with open(self.summary_stats_csv_path, "rb") as f_summary:
                            st.download_button(
                                label=f"Download Summary Statistics CSV",
                                data=f_summary,
                                file_name=os.path.basename(self.summary_stats_csv_path),
                                mime="text/csv",
                                key=f"download_summary_csv_{uuid.uuid4().hex}"
                            )
                    except Exception as e_read_summary:
                        st.error(f"Could not display saved summary statistics CSV: {e_read_summary}")
                        logging.error(f"Error reading summary CSV {self.summary_stats_csv_path}: {e_read_summary}", exc_info=True)
                else:
                    pass
            # else:
                    # logging.info(f"Data type {current_data_type} not eligible for automatic pandas summary CSV display.")
                    # self.summary_stats_csv_path = None # Ensure it's cleared if not applicable
            return True
        except Exception as e:
            st.error(f"Error previewing dataset: {e}")


    def save_chat_history(self):
        history = {
            "messages": st.session_state.get("messages", []),
            "eda_report": st.session_state.get("eda_report", ""),
            "memory": list(st.session_state.get("memory", [])),
            "feedback_submitted": { key: st.session_state[key] for key in st.session_state if key.startswith("feedback_submitted_") },
            "feedback_ids": { key: st.session_state[key] for key in st.session_state if key.startswith("feedback_id_") },
            "analysis_file_path": st.session_state.get("analysis_file_path", ""),
            "input_data_type": st.session_state.get("input_data_type", ""),
            "bookmarks": st.session_state.get("bookmarks", []),
            "summary_stats_csv_path": st.session_state.get("summary_stats_csv_path", ""),
        }
        with open(self.chat_hisory_file, "w") as f:
            json.dump(history, f, indent=2)
        bookmark_history = {
            "bookmarks": st.session_state.get("bookmarks", []),
        }
        with open("bookmarks.json", "w") as f:
            json.dump(bookmark_history, f, indent=2)


    def load_chat_history(self):
        if os.path.exists(self.chat_hisory_file):
            with open(self.chat_hisory_file, "r") as f:
                file_contents = f.read().strip()
                if file_contents:
                    history = json.loads(file_contents)
                    st.session_state["messages"] = history.get("messages", [])
                    st.session_state["eda_report"] = history.get("eda_report", "")
                    memory_list = history.get("memory", [])
                    st.session_state["memory"] = deque(memory_list, maxlen=15)
                    for key, value in history.get("feedback_submitted", {}).items():
                        st.session_state[key] = value
                    for key, value in history.get("feedback_ids", {}).items():
                        st.session_state[key] = value
                    st.session_state["analysis_file_path"] = history.get("analysis_file_path", "")
                    st.session_state["input_data_type"] = history.get("input_data_type", "")
                    st.session_state["bookmarks"] = history.get("bookmarks", [])
                    st.session_state["summary_stats_csv_path"] = history.get("summary_stats_csv_path", "")
                else:
                    # File is empty; initialize session state with defaults.
                    st.session_state["messages"] = []
                    st.session_state["eda_report"] = ""
                    st.session_state["memory"] = deque(maxlen=15)
                    st.session_state["bookmarks"] = []
                    st.session_state["summary_stats_csv_path"] = ""

    
    def load_dataset_preview(self):
        st.error("i am loading dataset preview")
        if "analysis_file_path" in st.session_state and st.session_state["analysis_file_path"]:
            return self.preview_dataset(st.session_state["analysis_file_path"])
        return None

    def format_memory_steps(self):
        middle_steps = ""
        # steps_list = []
        try:
            full_steps = self.agent.memory.get_full_steps()
            # st.write(full_steps)
            for idx, step in enumerate(full_steps, start=1):
                has_content = False
                for key in ["system_prompt", "observations", "model_output", "action_output", "error"]:
                    if key in step and step[key]:
                        if isinstance(step[key], str) and step[key].strip():
                            has_content = True
                        elif not isinstance(step[key], str):
                            has_content = True
                if not has_content:
                    continue
                step_number = step.get("step") or idx
                middle_steps += f"##### Step {step_number}:\n\n"
                
                if step.get("model_output"):
                    middle_steps += f"**Model Output**:\n\n {step['model_output']}\n\n"
                if step.get("action_output"):
                    middle_steps += f"**Action Output**:\n\n {step['action_output']}\n\n"
                if step.get("observations"):
                    middle_steps += f"**Observations**:\n\n {step['observations']}\n\n"
                if step.get("error"):
                    middle_steps += f"Error:\n\n {step['error']}\n\n"
                middle_steps += "\n\n"

                middle_steps = fix_code_block_formatting(middle_steps)
            
        except Exception as e:
            logging.error("Error retrieving memory steps: %s", e)
        return middle_steps


    def submit_feedback_response(self, feedback, msg_idx):
        if not st.session_state.get("db_available", False):
            st.warning("Feedback feature is disabled because the database is not connected.")
            return
        messages = st.session_state.get("messages", [])
        if msg_idx > 0 and messages[msg_idx - 1]["role"] == "user":
            question = messages[msg_idx - 1]["content"]
        else:
            question = "Unknown question"
        answer = messages[msg_idx]["content"]
        dataset_path = st.session_state.get("analysis_file_path", "")
        feedback_id = store_message_feedback(self.user_id, question, answer, feedback, dataset_path)
        st.session_state[f"feedback_submitted_{msg_idx}"] = True
        st.session_state[f"feedback_id_{msg_idx}"] = feedback_id
        self.save_chat_history()

    def display_middle_steps(self, steps_list):
        with st.expander("View Intermediate Steps"):
            for step in steps_list:
                st.markdown(f"##### Step {step['step_number']}:")
                for seg_type, seg_text in step["segments"]:
                    if seg_type == "text":
                        st.markdown(seg_text)
                    elif seg_type == "code":
                        st.code(seg_text)

    def display_chat_history(self):
        messages = st.session_state.get("messages", [])
        
        for idx, message in enumerate(messages):
            if not message or not message.get("role") or not message.get("content"):
                continue
            with st.chat_message(message["role"]):
                # Display the main content.
                if "count" in message.get("content", "") and "mean" in message.get("content", "") and "std" in message.get("content", ""):
                    st.code(message["content"])
                else:
                    st.markdown(message["content"])

                # Display candidate solutions if they exist.
                if "candidate_solutions" in message and message["candidate_solutions"]:
                    st.markdown("### Candidate Solutions")
                    for c_idx, candidate in enumerate(message["candidate_solutions"], start=1):
                        with st.expander(f"Candidate {c_idx}: {candidate.get('option', 'Option')}"):
                            st.markdown(f"**Explanation:** {candidate.get('explanation', '')}")
                            st.markdown(f"**Pros:** {candidate.get('pros', '')}")
                            st.markdown(f"**Cons:** {candidate.get('cons', '')}")
                            if st.button("Refine this solution", key=f"history_refine_candidate_{idx}_{c_idx}"):
                                prefill = candidate.get("option", "") + " " + candidate.get("explanation", "")
                                st.session_state["prefilled_input"] = prefill

                # Display intermediate steps if available.
                if "middle_steps" in message and message["middle_steps"]:
                    with st.expander("View Intermediate Steps"):
                        st.markdown(message["middle_steps"])

                # Display any generated plots.
                if "image_paths" in message:
                    for plot_path in message["image_paths"]:
                        if os.path.exists(plot_path):
                            image = Image.open(plot_path)
                            file_name = os.path.basename(plot_path)
                            file_name_no_ext = os.path.splitext(file_name)[0]
                            st.image(image, caption=file_name_no_ext)

                # Display file download buttons for any generated files.
                if "file_paths" in message:
                    for file_path in message["file_paths"]:
                        if os.path.exists(file_path):
                            
                            
                            if file_path.lower().endswith(".tsv"):
                                try:
                                    df = pd.read_csv(file_path, sep="\t")
                                    st.markdown(f"Preview of **{os.path.basename(file_path)}**:")
                                    st.dataframe(df)
                                except Exception as e:
                                    print(f"Error reading CSV file {os.path.basename(file_path)}: {e}")
                            
                            if file_path.lower().endswith(".csv"):
                                try:
                                    df = pd.read_csv(file_path)
                                    st.markdown(f"Preview of **{os.path.basename(file_path)}**:")
                                    st.dataframe(df)
                                except Exception as e:
                                    print(f"Error reading CSV file {os.path.basename(file_path)}: {e}")

                            unique_key = str(uuid.uuid4())
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {os.path.basename(file_path)}",
                                    data=f,
                                    file_name=os.path.basename(file_path),
                                    key=f"history_download_{unique_key}"
                                )
                
                
                if message["role"] == "assistant":
                # If feedback hasn't been submitted for this message, show the thumbs buttons.
                    if st.session_state.get("db_available", False):
                        if not st.session_state.get(f"feedback_submitted_{idx}", False):
                            col1, col2 = st.columns(2)
                            col1.button("👍", key=f"thumbs_up_{idx}", on_click=self.submit_feedback_response, args=("Yes", idx))
                            col2.button("👎", key=f"thumbs_down_{idx}", on_click=self.submit_feedback_response, args=("No", idx))
                            
                        else:
                            st.info("Feedback recorded!")
                            comment = st.text_area("Optional comment:", key=f"feedback_comment_{idx}")
                            if st.button("Update Comment", key=f"update_comment_{idx}"):
                                feedback_id = st.session_state.get(f"feedback_id_{idx}")
                                update_feedback_comment(feedback_id, comment)
                                st.success("Comment updated!")
                    
                    if not message.get("bookmarked", False):
                        # Grab the preceding user message if it exists, else leave blank
                        prev_q = (
                            messages[idx - 1]["content"]
                            if idx > 0 and messages[idx - 1]["role"] == "user"
                            else ""
                        )
                        bookmark_data = {
                            "question": prev_q,
                            "answer": message["content"],
                            "plots": message.get("image_paths", []),
                            "files": message.get("file_paths", [])
                        }
                        if st.button("🔖 Bookmark this response", key=f"bookmark_{idx}"):
                            st.session_state["bookmarks"].append(bookmark_data)
                            # mark in-place so button won’t reappear
                            st.session_state["messages"][idx]["bookmarked"] = True
                            self.save_chat_history()
                            st.rerun()
                            st.success("Response bookmarked!")
                    else:
                        st.markdown("✅ Bookmarked")
                
                # Display next steps suggestions.
                if "next_steps_suggestion" in message and message["next_steps_suggestion"] and idx != len(messages) - 1:
                    st.markdown(f"**Next Steps Suggestion:** \n* {message['next_steps_suggestion']}")
                            
        if messages:
            last_message = messages[-1]
            # Only display suggestion buttons if the last message is from the assistant and has suggestions
            if last_message["role"] == "assistant" and last_message.get("next_steps_suggestion") and not last_message.get("candidate_solutions"):
                suggestions = [s.strip() for s in last_message["next_steps_suggestion"].split("\n") if s.strip()]
                self.display_suggestion_buttons(suggestions)

    def display_suggestion_buttons(self, suggestions):
        """Display next step suggestions as clickable links inside the chat."""
        if not suggestions:
            return

        st.markdown("**Next Steps Suggestion:**")
        
        try:
            for idx, suggestion in enumerate(suggestions):
            # Make the text clickable and store the suggestion in session state
                if st.button(f"{suggestion}", key=f"suggestion_{suggestion.replace(' ', '_')}_{idx}"):
                    st.session_state["prefilled_input"] = suggestion
        except Exception as e:
            logging.error("Error displaying suggestion buttons: %s", e)

    def get_agent_prompt(self, dataset_path, user_question, question_type: int=2):
        
        memory_history = ""
        if st.session_state.get("memory"):
            memory_history = "\n".join(st.session_state["memory"])
        memory_history += "previous conversation ends here.\n\n"
        if question_type == 0:
            return (
                "You are an expert data analysis assistant who can solve any task using code blobs." 
                "To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.\n\n"
                f"```The dataset is saved at {dataset_path}. This is a {self.input_data_type} file. Please ignore the file extension of the file name, and use the dataset type: {self.input_data_type} to determine how to read the dataset. {user_question}. You must generate plots to answer this question!```\n\n"
                "- Always suggest possible next steps for data analysis at the end of the answer, unless the user is explicitly asking for suggestions.\n"
                f"- You should find an appropriate method to generate plots for this query. If a plot or file is generated, save it in the directory {self.output_dir} with a random numerical suffix to prevent overwrites.\n"
                "- Do not generate filenames like 'random_forest_model_XXXX.joblib'.\n"
                "- Always consider to generate plots or files to support your answer.\n"
                "- Always call the final_answer tool, providing the final answer in the following dictionary format (do not format as a JSON code block):\n"
                '{ "explanation": ["Your explanation here, in plain text. This can include detailed information or step-by-step guidance."], '
                '"plots": ["<path_to_the_image>" (leave the list empty if no plots are needed)], '
                '"files": ["<path_to_the_file>" (leave the list empty if no files are needed)], '
                '"next_steps_suggestion": ["List of possible next questions the user could ask to gain further insights. They should be questions. Only include this when the user has not explicitly asked for suggestions."] }'
            )
        elif question_type == 1:
            return (
                "You are an expert data analysis assistant who can solve any task using code blobs." 
                "To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.\n\n"
                f"Previous conversation:\n{memory_history}\n\n```Current Question:{user_question}```\n\n"
                "- Always suggest possible next steps for data analysis at the end of the answer, unless the user is explicitly asking for suggestions.\n"
                f"- If a plot or file is generated, save it in the {self.output_dir} directory with a random numerical suffix to prevent overwrites.\n"
                "- Do not generate filenames like 'random_forest_model_XXXX.joblib'.\n"
                "- Always consider to generate plots or files to support your answer.\n"
                "- Always call the final_answer tool, providing the final answer in the following dictionary format (do not format as a JSON code block):\n"
                '{ "explanation": ["Your explanation here, in plain text. This can include detailed information or step-by-step guidance."], '
                '"plots": ["<path_to_the_image>" (leave the list empty if no plots are needed)], '
                '"files": ["<path_to_the_file>" (leave the list empty if no files are needed)], '
                '"next_steps_suggestion": ["List of possible next questions the user could ask to gain further insights. They should be questions. Only include this when the user has not explicitly asked for suggestions."] }'
            )
        else:
            return (
                f"Previous conversation:\n{memory_history}\n\n"
                "You are an expert data analysis assistant who can solve any task using code blobs." 
                "To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences.\n\n"
                f"```The dataset is saved at {dataset_path}. This is a {self.input_data_type} file. Please ignore the file extension of the file name, and use the dataset type: {self.input_data_type} to determine how to read the dataset. Current Question: {user_question}```\n\n"
                "- Before answering, please analyze the user's question. If you determine the question is multifaceted, ambiguous, or covers several aspects, provide three distinct candidate solutions. For each candidate, include:\n"
                "   - An 'option' title,\n"
                "   - A detailed 'explanation',\n"
                "   - A list of 'pros',\n"
                "   - A list of 'cons'.\n"
                "- If the question is straightforward, provide a single concise answer following the standard format. But most questions should be strightforward.\n"
                "- Always include next step suggestions at the end.\n"
                f"- If a plot or file is generated, save it in the {self.output_dir} directory with a random numerical suffix to prevent overwrites.\n"
                "- Do not generate filenames like 'random_forest_model_XXXX.joblib'.\n"
                "- Always consider to generate plots or files to support your answer.\n"
                "- Always call the final_answer tool, providing the final answer in one of the following dictionary formats (do not format as a JSON code block):\n\n"
                "Simple answer format:\n"
                '{ "explanation": ["Your explanation text. in plain text. This can include detailed information or step-by-step guidance."], "plots": ["<path_to_image>" (leave the list empty if no plots are needed)], "files": ["<path_to_file>" (leave the list empty if no files are needed)], "next_steps_suggestion": ["Suggestion 1", "Suggestion 2"] }\n\n'
                "Multiple candidate solutions format:\n"
                '{ "candidate_solutions": [ { "option": "Solution 1", "explanation": "Detailed explanation...", "pros": "Pros...", "cons": "Cons..." }, { "option": "Solution 2", "explanation": "Detailed explanation...", "pros": "Pros...", "cons": "Cons..." }, { "option": "Solution 3", "explanation": "Detailed explanation...", "pros": "Pros...", "cons": "Cons..." } ], "next_steps_suggestion": ["Which option would you like to refine?", "Or ask for more details on a candidate solution."] }'
            )

    def handle_user_input(self, temp_file_path, user_question, question_type=2):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state["messages"].append({"role": "user", "content": user_question})
        st.session_state["memory"].append(f"User: {user_question}")
        self.save_chat_history()
        # self.display_chat_history()
        with st.spinner("Thinking..."):
            prompt = self.get_agent_prompt(temp_file_path, user_question, question_type=question_type)
            response = self.agent.run(prompt)
            # print(f"after agent.run: {self.agent.monitor.get_total_token_counts()}")
            middle_steps = self.format_memory_steps()
            self.process_response(response, middle_steps)
        self.save_chat_history()

    def run_eda(self, temp_file_path):
        eda_query = (
            "Perform a comprehensive exploratory data analysis (EDA) on the provided dataset. "
            "Answer these questions one by one:  "
            "What are the summary statistics for the dataset?"
            "What are the missing values in the dataset?"
            "Show the correlation matrix of the features."
            "Show the distribution of numerical features."
            "Any insights?"
        )
        eda_display_query = "Perform a comprehensive exploratory data analysis (EDA) on the provided dataset."
        with st.chat_message("user"):
            st.markdown(eda_display_query)
        st.session_state["messages"].append({"role": "user", "content": eda_display_query})
        st.session_state["memory"].append(f"User: {eda_display_query}")
        with st.spinner("Running EDA..."):
            try:
                eda_response = self.agent.run(self.get_agent_prompt(temp_file_path, eda_query, question_type=0))
                parsed = self.parse_response_content(eda_response)
                middle_steps = self.format_memory_steps()
                if parsed and parsed.get("explanation"):
                    report_text = "\n".join(parsed["explanation"])
                else:
                    report_text = "Try click the button again to run EDA."
                # report_text = clean_text(report_text)
                
                html_content = "<html><head><title>EDA Report</title></head><body>"
                html_content += "<h1>Exploratory Data Analysis Report</h1>"
                html_content += f"<h2>Report Summary</h2><p>{report_text.replace(chr(10), '<br>')}</p>"
                if parsed and parsed.get("plots"):
                    html_content += "<h2>Visualizations</h2>"
                    for plot_path in parsed["plots"]:
                        if os.path.exists(plot_path):
                            with open(plot_path, "rb") as img_file:
                                encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
                            html_content += f'<div><img src="data:image/png;base64,{encoded_string}" style="max-width:600px;"></div><br/>'
                if parsed and parsed.get("next_steps_suggestion"):
                    html_content += "<h2>Next Steps Suggestions</h2><ul>"
                    for suggestion in parsed["next_steps_suggestion"]:
                        html_content += f"<li>{suggestion}</li>"
                    html_content += "</ul>"
                html_content += "</body></html>"

                # Save the EDA report to an HTML file.
                eda_file_path = os.path.join(self.output_dir, "eda_report.html")
                with open(eda_file_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                st.session_state["eda_report"] = eda_file_path

                # st.success("EDA complete! Download the report below:")
                # st.download_button(
                #     label="Download EDA Report",
                #     data=html_content,
                #     file_name="eda_report.html",
                #     mime="text/html"
                # )

                file_paths = parsed.get("files", [])
                file_paths = [eda_file_path] + file_paths if file_paths else [eda_file_path]

                eda_result_message = {
                    "role": "assistant",
                    "content": report_text,
                    "image_paths": parsed.get("plots", []) if parsed else [],
                    "file_paths": file_paths,
                    "next_steps_suggestion": "  \n* ".join(parsed.get("next_steps_suggestion", [])) if parsed else "",
                    "middle_steps": middle_steps
                }
                st.session_state["messages"].append(eda_result_message)
                st.session_state["memory"].append(f"Assistant (EDA): {report_text}")

                self.save_chat_history()
                st.rerun()
            except Exception as e:
                st.error(f"Error during EDA: {e}")

    def summarize_chat_history(self):
        summary_prompt = "Summarize the previous conversation in a concise manner.\n"
        self.handle_user_input("", summary_prompt, question_type=1)

    def parse_response_content(self, content):
        if not isinstance(content, str):
            return content
        try:
            content_no_comments = re.sub(r'//.*', '', content)
            parsed = json.loads(content_no_comments)
            if "candidate_solutions" in parsed:
                return {
                    "candidate_solutions": parsed["candidate_solutions"],
                    "next_steps_suggestion": parsed.get("next_steps_suggestion", [])
                }
            else:
                
                message = {
                    "explanation": "\n".join(parsed.get("explanation", [])),
                    "plots": parsed.get("plots", []),
                    "files": parsed.get("files", []),
                    "next_steps_suggestion": "  \n* ".join(parsed.get("next_steps_suggestion", []))
                }
                return message
        except json.JSONDecodeError as e:
            logging.error("JSON decode error: %s", e)
            return None

    def process_response(self, response, middle_steps=""):
        # Case 1: Response is an object with 'role' and 'content' attributes.
        if hasattr(response, 'role') and hasattr(response, 'content'):
            role = getattr(response, 'role', 'assistant')
            content = getattr(response, 'content', '')
            parsed_message = self.parse_response_content(content)
            if parsed_message:
                if "candidate_solutions" in parsed_message:
                    candidate_list = parsed_message["candidate_solutions"]
                    next_steps = "  \n* ".join(parsed_message.get("next_steps_suggestion", []))
                    st.session_state["messages"].append({
                        "role": role,
                        "content": "Multiple candidate solutions generated.",
                        "candidate_solutions": candidate_list,
                        "image_paths": [],
                        "file_paths": [],
                        "next_steps_suggestion": next_steps,
                        "middle_steps": middle_steps
                    })
                    st.session_state["memory"].append(f"{role.capitalize()}: Multiple candidate solutions generated.")
                    
                else:
                    message = {
                        "explanation": parsed_message.get("explanation", ""),
                        "plots": parsed_message.get("plots", []),
                        "files": parsed_message.get("files", []),
                        "next_steps_suggestion": parsed_message.get("next_steps_suggestion", ""),
                        "middle_steps": middle_steps
                    }
                    if not message["plots"] and not message["files"]:
                        message["explanation"] += "\nLLM did not generate any plots or files."
                    st.session_state["messages"].append({
                        "role": role,
                        "content": message["explanation"],
                        "image_paths": message["plots"],
                        "file_paths": message["files"],
                        "next_steps_suggestion": message["next_steps_suggestion"],
                        "middle_steps": message["middle_steps"]
                    })
                    st.session_state["memory"].append(f"{role.capitalize()}: {message['explanation']}")
                    
            else:
                st.session_state["messages"].append({
                    "role": role,
                    "content": f"Response received:\n\n{content}\n"
                })

        # Case 2: Response is a dictionary.
        elif isinstance(response, dict):
            if "candidate_solutions" in response:
                candidate_list = response["candidate_solutions"]
                next_steps = "  \n* ".join(response.get("next_steps_suggestion", []))
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": "Multiple candidate solutions generated.",
                    "candidate_solutions": candidate_list,
                    "image_paths": [],
                    "file_paths": [],
                    "next_steps_suggestion": next_steps,
                    "middle_steps": middle_steps
                })
                st.session_state["memory"].append("Assistant: Multiple candidate solutions generated.")
                
            else:
                message = {
                    "explanation": "\n".join(response.get("explanation", [])),
                    "plots": response.get("plots", []),
                    "files": response.get("files", []),
                    "next_steps_suggestion":  "  \n* ".join(response.get("next_steps_suggestion", [])),
                    "middle_steps": middle_steps
                }
                # st.markdown(message["explanation"])
                if not message["plots"] and not message["files"]:
                    message["explanation"] += "\nLLM did not generate any plots or files."
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": message["explanation"],
                    "image_paths": message["plots"],
                    "file_paths": message["files"],
                    "next_steps_suggestion": message["next_steps_suggestion"],
                    "middle_steps": message["middle_steps"]
                })
                st.session_state["memory"].append("Assistant: " + message["explanation"])
                

        # Case 3: Response is a plain string.
        elif isinstance(response, str):
            parsed_message = self.parse_response_content(response)
            if parsed_message:
                self.process_response(parsed_message, middle_steps)
            else:
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": f"Response received:\n\n{response}\n"
                })

        # Fallback: Any other type.
        else:
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"Response received:\n\n{response}\n"
            })
        self.save_chat_history()
        st.rerun()



    def has_eda_history(self):
        if "eda_report" in st.session_state and st.session_state["eda_report"]:
            return True
        return False

    def run(self):
  
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            self.load_chat_history()
        if "file_paths" not in st.session_state:
            st.session_state["file_paths"] = []

        if "analysis_file_path" not in st.session_state:
            st.session_state["analysis_file_path"] = ""
        if "eda_report" not in st.session_state:
            st.session_state["eda_report"] = ""

        # Load existing EDA report if it exists.
        eda_path = os.path.join(self.output_dir, "eda_report.html")
        if os.path.exists(eda_path):
            st.session_state["eda_report"] = eda_path
        
        # Determine which dataset to use.
        uploaded_file = None
        if not self.dataset_file_path:
            uploaded_file = st.file_uploader("Upload your dataset (CSV or TSV)", type=["csv", "tsv", "h5ad", "xlsx", "xls", "json", "parquet", "h5", "fa", "fasta", "vcf", "gtf", "gff", "bed"])
        df = None
        if uploaded_file or self.dataset_file_path:
            try:
                output_file_path = None
                if uploaded_file:
                    output_file_path = os.path.join(self.output_dir, uploaded_file.name)
                    with open(output_file_path, "wb") as out_file:
                        import shutil
                        shutil.copyfileobj(uploaded_file, out_file)
                elif self.dataset_file_path:
                    output_file_path = self.dataset_file_path
                df = self.preview_dataset(output_file_path)
                st.session_state["analysis_file_path"] = output_file_path
                self.save_chat_history()
                st.session_state["eda_report"] = None
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        elif st.session_state["analysis_file_path"]:
            df = self.load_dataset_preview()

        if df is not None:
            st.success("Dataset loaded successfully!")
            # st.write("Preview of your dataset:")
            # st.dataframe(df)
            self.display_chat_history()

            # temp_file_path = os.path.join(self.output_dir, "temp_data.csv")
            # st.session_state["temp_file_path"] = temp_file_path
            # df.to_csv(temp_file_path, index=False)
            # if uploaded_file is not None:
            #     df.to_csv("uploaded_dataset.csv", index=False)
            
            user_question = st.chat_input("Ask a question about the dataset")
            if user_question or st.session_state.get("prefilled_input"):
                if st.session_state.get("prefilled_input"):
                    user_question = st.session_state["prefilled_input"]
                st.session_state["prefilled_input"] = None
                self.handle_user_input(st.session_state["analysis_file_path"] , user_question)

            st.sidebar.markdown("---")
            st.sidebar.markdown("### Exploratory Data Analysis")
            if st.session_state.get("eda_report") and os.path.exists(st.session_state["eda_report"]):
                with open(st.session_state["eda_report"], "rb") as f:
                    html_content = f.read()
                st.sidebar.download_button(
                    label="Download EDA Report",
                    data=html_content,
                    key="eda_report_sidebar",
                    file_name="eda_report.html",
                    mime="text/html"
                )

            if st.sidebar.button("Run EDA", key="run_eda"):
                self.run_eda(st.session_state["analysis_file_path"])
            # elif not self.has_eda_history():
            #     self.run_eda(temp_file_path)

            st.write("You can now interact with the chatbot to ask questions about the dataset.")
            
            if os.path.exists(st.session_state["analysis_file_path"]):
                if st.sidebar.button("Correlation Matrix", key="corr_matrix"):
                    self.handle_user_input(st.session_state["analysis_file_path"], "Show the correlation matrix of the features.")
                if st.sidebar.button("Missing Values", key="missing_values"):
                    self.handle_user_input(st.session_state["analysis_file_path"], "What are the missing values in the dataset?")
                if st.sidebar.button("Numerical Feature Distribution", key="num_dist"):
                    self.handle_user_input(st.session_state["analysis_file_path"], "Show the distribution of numerical features.")
                # if st.sidebar.button("Summary Statistics", key="summary_stats"):
                #     self.handle_user_input(st.session_state["analysis_file_path"], "What are the summary statistics for the dataset? return a csv file containing the summary statistics.")

            st.sidebar.markdown("---")
            st.sidebar.markdown("### Summarize Chat History")
            if st.sidebar.button("Summarize Chat", key="summarize_chat"):
                self.summarize_chat_history()
            
        else:
            st.info("Please upload a dataset.")


def main():

    print(sys.argv[:])

    user_id = sys.argv[1] if len(sys.argv) > 1 else None
    cli_openai_key_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    cli_groq_key_file_path = sys.argv[3] if len(sys.argv) > 3 else None
    chat_history_path = sys.argv[4] if len(sys.argv) > 4 else None
    generate_file_path = sys.argv[5] if len(sys.argv) > 5 else None
    input_file_path = sys.argv[6] if len(sys.argv) > 6 else None
    input_data_type = sys.argv[7] if len(sys.argv) > 7 else None

    if not user_id:
        st.error("No user ID provided. Please provide a user ID as a command-line argument.")
        return
    
    # openai_api_key = None
    # if openai_api_key_file:
    #     with open(openai_api_key_file, "r") as f:
    #         openai_api_key = f.read().strip()
    #     st.session_state["openai_api_key"] = openai_api_key
    # groq_api_key = None
    # if groq_api_key_file:
    #     with open(groq_api_key_file, "r") as f:
    #         groq_api_key = f.read().strip()
    #     st.session_state["groq_api_key"] = groq_api_key

    if chat_history_path:
        st.session_state["chat_history_path"] = chat_history_path
    if generate_file_path:
        st.session_state["generate_file_path"] = generate_file_path
    if input_file_path:
        st.session_state["input_file_path"] = input_file_path
    if input_data_type:
        st.session_state["input_data_type"] = input_data_type

    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = "" # Initialize as empty string
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = "" 
    
    if cli_openai_key_file_path:
        logging.info(f"Attempting to load OpenAI key from CLI file: {cli_openai_key_file_path}")
        cli_openai_key = load_key_from_specific_file(cli_openai_key_file_path)
        if cli_openai_key:
            st.session_state.openai_api_key = cli_openai_key
            save_key_to_specific_file(OPENAI_API_KEY_FILE, cli_openai_key)
            logging.info("OpenAI key loaded from CLI file and also saved to local config.")
    
    # 2. If not loaded from CLI (or if CLI key was empty), try to load OpenAI key from local config file
    if not st.session_state.openai_api_key: 
        loaded_key = load_key_from_specific_file(OPENAI_API_KEY_FILE)
        if loaded_key:
            st.session_state.openai_api_key = loaded_key
            logging.info("OpenAI key loaded from local config file.")

    # Repeat for Groq API Key
    if cli_groq_key_file_path:
        logging.info(f"Attempting to load Groq key from CLI file: {cli_groq_key_file_path}")
        cli_groq_key = load_key_from_specific_file(cli_groq_key_file_path)
        if cli_groq_key:
            st.session_state.groq_api_key = cli_groq_key
            save_key_to_specific_file(GROQ_API_KEY_FILE, cli_groq_key)
            logging.info("Groq key loaded from CLI file and also saved to local config.")

    if not st.session_state.groq_api_key: 
        loaded_key = load_key_from_specific_file(GROQ_API_KEY_FILE)
        if loaded_key:
            st.session_state.groq_api_key = loaded_key
            logging.info("Groq key loaded from local config file.")


    try:
        init_feedback_db()
    except Exception as e:
        logging.error("Could not initialize feedback DB: %s", e)

    st.session_state["db_available"] = check_db_connection()
    
    st.title("Data Analysis Agent")
    st.sidebar.title("Configuration")
    MODEL_OPTIONS = {
        "OpenAI (GPT-4o)": "gpt-4o",
        "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
        "OpenAI (GPT-4)": "gpt-4",
        "OpenAI (GPT-3.5-Turbo)": "gpt-3.5-turbo",
        "Groq (Llama-3.3-70B-Versatile)": "llama-3.3-70b-versatile",
        "Groq (llama3-70b-8192)": "llama3-70b-8192",
    }
    model_keys = list(MODEL_OPTIONS.keys())
    selected_model_name = st.sidebar.selectbox("Select LLM Model", model_keys, index=0)
    selected_model = MODEL_OPTIONS[selected_model_name]
    st.session_state["selected_model"] = selected_model
    is_openai_selected = selected_model.startswith("gpt-") or selected_model == "gpt-4o"
    is_groq_selected = selected_model.startswith("llama-")
    # if is_openai and not openai_api_key:
    #     st.sidebar.markdown(f"getting here: {not openai_api_key}")
    #     openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
    #     st.session_state["openai_api_key"] = openai_api_key
    # elif is_groq and not groq_api_key:
    #     groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
    #     st.session_state["selected_model"] = "groq/" + st.session_state["selected_model"]
    #     st.session_state["groq_api_key"] = groq_api_key

    api_key_action_taken = False 

    if is_openai_selected:
        if not st.session_state.get("openai_api_key"): 
            st.sidebar.markdown("---")
            st.sidebar.subheader("OpenAI API Key Required")
            widget_openai_key_input = st.sidebar.text_input(
                "Enter your OpenAI API Key:", type="password", key="widget_openai_key_input_field_v3"
            )
            if st.sidebar.button("Save and Apply OpenAI Key", key="save_openai_button_v3"):
                if widget_openai_key_input:
                    st.session_state.openai_api_key = widget_openai_key_input
                    save_key_to_specific_file(OPENAI_API_KEY_FILE, widget_openai_key_input)
                    logging.info("OpenAI Key saved from UI input.")
                    api_key_action_taken = True
                else:
                    st.sidebar.error("API Key cannot be empty.")
        else:
            st.sidebar.success(f"OpenAI API Key is configured.")
            if st.sidebar.button("Clear/Change OpenAI Key", key="clear_openai_button_v3"):
                save_key_to_specific_file(OPENAI_API_KEY_FILE, "") 
                st.session_state.openai_api_key = "" 
                api_key_action_taken = True 

    elif is_groq_selected:
        if not st.session_state.get("groq_api_key"): 
            st.sidebar.markdown("---")
            st.sidebar.subheader("Groq API Key Required")
            widget_groq_key_input = st.sidebar.text_input(
                "Enter your Groq API Key:", type="password", key="widget_groq_key_input_field_v3"
            )
            if st.sidebar.button("Save and Apply Groq Key", key="save_groq_button_v3"):
                if widget_groq_key_input:
                    st.session_state.groq_api_key = widget_groq_key_input
                    save_key_to_specific_file(GROQ_API_KEY_FILE, widget_groq_key_input)
                    logging.info("Groq Key saved from UI input.")
                    api_key_action_taken = True
                else:
                    st.sidebar.error("API Key cannot be empty.")
        else:
            st.sidebar.success(f"Groq API Key is configured.")
            if st.sidebar.button("Clear/Change Groq Key", key="clear_groq_button_v3"):
                save_key_to_specific_file(GROQ_API_KEY_FILE, "") 
                st.session_state.groq_api_key = "" 
                api_key_action_taken = True 

    if api_key_action_taken:
        st.rerun() 

    # --- Determine final API key FOR THE AGENT ---
    final_api_key_for_agent = None
    final_model_id_for_agent = selected_model

    if is_openai_selected:
        final_api_key_for_agent = st.session_state.get("openai_api_key")
    elif is_groq_selected:
        final_api_key_for_agent = st.session_state.get("groq_api_key")
        # LiteLLM convention for Groq models often requires prefixing "groq/"
        if final_model_id_for_agent and not final_model_id_for_agent.startswith("groq/"):
            final_model_id_for_agent = "groq/" + final_model_id_for_agent

    # --- Agent Initialization and App Run ---

    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"] div.stButton > button {
            width: 100%;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        div[data-testid="stDownloadButton"] button {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    if final_api_key_for_agent and final_model_id_for_agent:
        agent = create_agent(final_api_key_for_agent, final_model_id_for_agent)
        app = StreamlitApp(agent=agent,
                           user_id=user_id,
                           output_dir=st.session_state["generate_file_path"],
                           dataset_file_path=st.session_state["input_file_path"],
                           chat_history_file=st.session_state["chat_history_path"],
                           input_data_type=st.session_state["input_data_type"])
        app.run()
    else:
        st.sidebar.warning("Please enter the required API Key to use the app.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Error in main: %s", e)
        st.error(f"An error occurred: {e}")
        st.error("Please try again.")
    
