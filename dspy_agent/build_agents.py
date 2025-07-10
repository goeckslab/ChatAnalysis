import os
import dspy
import cloudpickle as pickle
from pathlib import Path
import argparse
import logging

# Import necessary components from your other files
from agent_definitions import (
    DataAnalysisAgentModule,
    load_examples_from_json,
    validation_metric,
)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
SCRIPT_PATH = Path(__file__).resolve().parent
DEFAULT_DSPY_EXAMPLES_FILE = SCRIPT_PATH / "examples.json"
COMPILED_AGENTS_DIR = SCRIPT_PATH / "compiled_agents"

def compile_and_save_agent(model_id: str, api_key: str | None, examples_path: Path):
    """Compiles a DSPy agent for a given model and saves it to a file."""
    
    provider = model_id.split('/')[0]
    model_name = model_id.split('/')[-1].replace('.', '_') # Sanitize filename
    output_filename = f"agent_{provider}_{model_name}.pkl"
    output_path = COMPILED_AGENTS_DIR / output_filename
    
    logging.info(f"Starting compilation for model: {model_id}")
    
    # 1. Configure the Language Model
    try:
        if provider == "google":
            lm = dspy.LM(model_id) # Uses environment variables for API key
        else:
            if not api_key:
                raise ValueError(f"API key is required for provider '{provider}'")
            lm = dspy.LM(model_id, api_key=api_key)
        
        dspy.settings.configure(lm=lm)
        logging.info("DSPy LM configured successfully.")
        
    except Exception as e:
        logging.error(f"Failed to configure LM for {model_id}: {e}")
        return

    # 2. Load training examples
    examples = load_examples_from_json(examples_path)
    if not examples:
        logging.warning(f"No examples found at {examples_path}. Agent will be uncompiled.")
        # Save uncompiled agent if you still want a file
        agent = DataAnalysisAgentModule(outputs_dir=Path("."), current_dataset_path=None)
    else:
        # 3. Define the teleprompter and compile
        teleprompter = dspy.BootstrapFewShot(
            metric=validation_metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=min(len(examples), 4),
        )
        
        # We pass dummy paths here because the real paths will be set at runtime in the main app
        uncompiled_agent = DataAnalysisAgentModule(outputs_dir=Path("."), current_dataset_path=None)
        
        try:
            agent = teleprompter.compile(uncompiled_agent, trainset=examples)
            logging.info(f"Successfully compiled agent for {model_id}")
        except Exception as e:
            logging.error(f"Compilation failed for {model_id}: {e}")
            return

    # 4. Save the compiled agent using cloudpickle
    COMPILED_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(agent, f)
    
    logging.info(f"Agent saved successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and save DSPy agents.")
    parser.add_argument("--model", required=True, help="Model ID to compile (e.g., 'openai/gpt-4o', 'google/gemini-2.5-pro').")
    parser.add_argument("--api_key", help="API key for the model provider (not needed for Google models).")
    parser.add_argument("--examples", default=str(DEFAULT_DSPY_EXAMPLES_FILE), help="Path to the DSPy examples JSON file.")
    
    args = parser.parse_args()
    
    compile_and_save_agent(
        model_id=args.model,
        api_key=args.api_key,
        examples_path=Path(args.examples)
    )