# build_agents.py

import os
import dspy
import cloudpickle as pickle
from pathlib import Path
import argparse
import logging

from chat_dspy import (
    DataAnalysisAgentModule,
    load_examples_from_json,
    validation_metric,
    MODEL_OPTIONS_SELECT
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_PATH = Path(__file__).resolve().parent
DEFAULT_DSPY_EXAMPLES_FILE = SCRIPT_PATH / "examples.json"
COMPILED_AGENTS_DIR = SCRIPT_PATH / "compiled_agents"

def compile_and_save_agent(model_id: str, api_key: str | None, examples_path: Path):
    """
    Compiles a DSPy agent for a given model and saves it to a file.
    """
    
    provider = model_id.split('/')[0]
    # Sanitize the model name to create a valid filename
    model_name_sanitized = model_id.split('/')[-1].replace('.', '_')
    output_filename = f"agent_{provider}_{model_name_sanitized}.pkl"
    output_path = COMPILED_AGENTS_DIR / output_filename
    
    logging.info(f"--- Starting compilation for model: {model_id} ---")
    
    # 1. Configure the Language Model using the unified dspy.LM
    try:
        # dspy.LM handles API keys for all supported providers.
        # It will use the provided key or fall back to environment variables if the key is None.
        lm = dspy.LM(model_id, api_key=api_key)
        dspy.settings.configure(lm=lm)
        logging.info(f"Configured DSPy LM for {model_id}.")
    except Exception as e:
        logging.error(f"Failed to configure LM for {model_id}: {e}")
        return

    # 2. Load training examples
    examples = load_examples_from_json(examples_path)
    if not examples:
        logging.warning(f"No examples found at {examples_path}. Agent for {model_id} will be saved UNCOMPILED.")
        # If no examples, we'll just save the uncompiled agent module
        agent = DataAnalysisAgentModule(outputs_dir=Path("."), current_dataset_path=None)
    else:
        # 3. Define the teleprompter and compile the agent
        teleprompter = dspy.BootstrapFewShot(metric=validation_metric, max_bootstrapped_demos=2)
        
        # We pass dummy paths for outputs_dir and dataset_path during compilation.
        # These will be updated with real paths at runtime in the main application.
        uncompiled_agent = DataAnalysisAgentModule(outputs_dir=Path("."), current_dataset_path=None)
        
        try:
            agent = teleprompter.compile(uncompiled_agent, trainset=examples)
            logging.info(f"Successfully compiled agent for {model_id}")
        except Exception as e:
            logging.error(f"Compilation failed for {model_id}: {e}")
            return

    # 4. Save the final agent (compiled or uncompiled) to a .pkl file
    COMPILED_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(agent, f)
    
    logging.info(f"Agent for {model_id} saved successfully to: {output_path}\n")

if __name__ == "__main__":
    # Set up argument parser to accept API keys from the command line
    parser = argparse.ArgumentParser(description="Compile and save all defined DSPy agents.")
    parser.add_argument("--openai_key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key.")
    parser.add_argument("--google_key", default=os.getenv("GOOGLE_API_KEY"), help="Google API key for Gemini models.")
    parser.add_argument("--groq_key", default=os.getenv("GROQ_API_KEY"), help="Groq API key.")
    parser.add_argument("--examples", default=str(DEFAULT_DSPY_EXAMPLES_FILE), help="Path to DSPy examples JSON file.")
    
    args = parser.parse_args()

    # Create a dictionary to map provider names to their respective keys
    api_keys = {
        "openai": args.openai_key,
        "gemini": args.google_key,
        "google": args.google_key,  # Handles both "gemini/" and "google/" prefixes
        "groq": args.groq_key,
    }
    
    # Loop through all models defined in the shared dictionary from agent_definitions.py
    for model_id in MODEL_OPTIONS_SELECT.keys():
        provider = model_id.split('/')[0]
        
        # Select the appropriate API key for the current model's provider
        api_key_for_model = api_keys.get(provider)
        
        # Some providers might not require a key if auth is handled differently (e.g., some cloud environments).
        # However, we'll warn if a key is expected but not found.
        if not api_key_for_model and provider in api_keys:
             logging.warning(f"API key for provider '{provider}' not found. Check environment variables or command-line args. Skipping {model_id}.")
             continue
        
        compile_and_save_agent(
            model_id=model_id,
            api_key=api_key_for_model,
            examples_path=Path(args.examples)
        )
    
    logging.info("--- All compilations finished. ---")