import os
import re
import pandas as pd
from PIL import Image
from collections import deque
from dotenv import load_dotenv
import streamlit as st
from smolagents import CodeAgent, LiteLLMModel
import json
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

@st.cache_resource
def create_agent(api_key, model_id):
    model = LiteLLMModel(model_id=model_id, api_key=api_key)
    return CodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=[
            "pandas", "numpy", "matplotlib", "PIL", "seaborn", 
            "sklearn", "pycaret", "plotly", "joblib", "io"
        ],
    )

class StreamlitApp:
    def __init__(self, agent):
        self.agent = agent
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        if "memory" not in st.session_state:
            st.session_state["memory"] = deque(maxlen=15)

    def load_dataset(self, file):
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.endswith(".tsv"):
            return pd.read_csv(file, sep="\t")
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or TSV file.")

    def save_chat_history(self):
        with open("chat_history.json", "w") as f:
            json.dump(st.session_state["messages"], f)

    def load_chat_history(self):
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as f:
                st.session_state["messages"] = json.load(f)

    def load_dataset_preview(self):
        if os.path.exists("uploaded_dataset.csv"):
            return pd.read_csv("uploaded_dataset.csv")
        return None

    def display_response(self, explanation, plot_paths, file_paths, next_steps_suggestion):
        """Helper function to display response content."""
        with st.chat_message("assistant"):
            if explanation:
                st.markdown(explanation)
            for plot_path in plot_paths:
                if plot_path and os.path.exists(plot_path):
                    image = Image.open(plot_path)
                    st.image(image, caption="Generated Plot")
            for file_path in file_paths:
                if file_path and os.path.exists(file_path):
                    unique_key = str(uuid.uuid4())
                    with open(file_path, "rb") as f:
                        st.download_button(
                            label=f"Download {os.path.basename(file_path)}",
                            data=f,
                            file_name=os.path.basename(file_path),
                            key=f"download_{unique_key}"
                        )
            if next_steps_suggestion:
                st.markdown(f"**Next Steps Suggestion:**  \n* {next_steps_suggestion}")
                st.markdown("Please let me know if you want to proceed with any of the suggestions or ask any other questions.")

    def display_chat_history(self):
        """Extracted function to display previous chat messages."""
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "image_paths" in message:
                    for plot_path in message["image_paths"]:
                        if os.path.exists(plot_path):
                            image = Image.open(plot_path)
                            st.image(image, caption="Generated Plot")
                if "file_paths" in message:
                    for file_path in message["file_paths"]:
                        if os.path.exists(file_path):
                            unique_key = str(uuid.uuid4())
                            with open(file_path, "rb") as f:
                                st.download_button(
                                    label=f"Download {os.path.basename(file_path)}",
                                    data=f,
                                    file_name=os.path.basename(file_path),
                                    key=f"history_download_{unique_key}"
                                )
                if "next_steps_suggestion" in message and message["next_steps_suggestion"]:
                    st.markdown(f"**Next Steps Suggestion:**  \n* {message['next_steps_suggestion']}")

    def get_agent_prompt(self, temp_file_path, user_question):
        """Helper to generate the prompt for the agent by prepending previous conversation history."""
        memory_history = ""
        if st.session_state.get("memory"):
            memory_history = "\n".join(st.session_state["memory"])
        return f"""
Previous conversation:
{memory_history}

The dataset is saved at {temp_file_path}. {user_question}

- Always suggest possible next steps for data analysis at the end of the answer, unless the user is explicitly asking for suggestions. If the user is asking for suggestions in the question, do not return `next_steps_suggestion`.
- You do not need to generate a plot every time. However, if a plot or file is generated, save it in the `outputs/` directory with a random numerical suffix in the filename to prevent overwriting files with the same name.
- Do not generate filenames like 'random_forest_model_XXXX.joblib'.
- Always call the `final_answer` tool, providing the final answer in the following dictionary format (do not format it as a JSON code block):

{{
    "explanation": ["Your explanation here, in plain text. This can include detailed information or step-by-step guidance."],
    "plots": ["<path_to_the_image>" (leave empty if no plots are needed)],
    "files": ["<path_to_the_file>" (leave empty if no files are needed)],
    "next_steps_suggestion": ["List of possible next questions the user could ask to gain further insights. Only include this when the user has not explicitly asked for suggestions."]
}}
        """

    def handle_user_input(self, temp_file_path):
        """Extracted function to handle user input and agent response."""
        if user_question := st.chat_input("Ask a question about the dataset"):
            # Append user question to both messages and memory
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state["messages"].append({
                "role": "user", 
                "content": user_question
            })
            st.session_state["memory"].append(f"User: {user_question}")

            with st.spinner("Generating response..."):
                prompt = self.get_agent_prompt(temp_file_path, user_question)
                response = self.agent.run(prompt)
            self.process_response(response)
            self.save_chat_history()

    def parse_response_content(self, content):
        """Helper function to parse the agent's response content into a standardized message format.
           This version removes inline comments (which are not allowed in JSON) before parsing."""
        try:
            # Remove any inline comments starting with // (be cautious: this simple regex may remove text in strings)
            content_no_comments = re.sub(r'//.*', '', content)
            parsed = json.loads(content_no_comments)
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

    def process_response(self, response):
        """Enhanced process_response with improved error handling and consistent message formatting."""
        if isinstance(response, dict):
            message = {
                "explanation": "\n".join(response.get("explanation", [])),
                "plots": response.get("plots", []),
                "files": response.get("files", []),
                "next_steps_suggestion": "  \n* ".join(response.get("next_steps_suggestion", []))
            }
            if not message["plots"] and not message["files"]:
                message["explanation"] += "\nLLM did not generate any plots or files."
            self.display_response(
                message["explanation"], 
                message["plots"], 
                message["files"], 
                message["next_steps_suggestion"]
            )
            st.session_state["messages"].append({
                "role": "assistant",
                "content": message["explanation"],
                "image_paths": message["plots"],
                "file_paths": message["files"],
                "next_steps_suggestion": message["next_steps_suggestion"],
            })
            st.session_state["memory"].append(f"Assistant: {message['explanation']}")
        elif isinstance(response, str):
            parsed_message = self.parse_response_content(response)
            if parsed_message:
                self.process_response(parsed_message)
            else:
                st.session_state["messages"].append({
                    "role": "assistant", 
                    "content": f"Response received:\n```json\n{response}\n```"
                })
        elif hasattr(response, 'role') and hasattr(response, 'content'):
            role = getattr(response, 'role', 'assistant')
            content = getattr(response, 'content', '')
            parsed_message = self.parse_response_content(content)
            if parsed_message:
                message = parsed_message
                if not message["plots"] and not message["files"]:
                    message["explanation"] += "\nLLM did not generate any plots or files."
            else:
                message = {
                    "explanation": content + "\nLLM did not generate any plots or files.",
                    "plots": [],
                    "files": [],
                    "next_steps_suggestion": ""
                }
            self.display_response(
                message["explanation"], 
                message["plots"], 
                message["files"], 
                message["next_steps_suggestion"]
            )
            st.session_state["messages"].append({
                "role": role,
                "content": message["explanation"],
                "image_paths": message["plots"],
                "file_paths": message["files"],
                "next_steps_suggestion": message["next_steps_suggestion"]
            })
            st.session_state["memory"].append(f"{role.capitalize()}: {message['explanation']}")
        else:
            st.session_state["messages"].append({
                "role": "assistant", 
                "content": f"Response received:\n```json\n{response}\n```"
            })

    def run(self):
        st.title("Data Analysis Agent")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            self.load_chat_history()

        if "file_paths" not in st.session_state:
            st.session_state["file_paths"] = []

        dataset = self.load_dataset_preview()
        uploaded_file = st.file_uploader("Upload your dataset (CSV or TSV)", type=["csv", "tsv"])

        if uploaded_file is not None:
            try:
                df = self.load_dataset(uploaded_file)
                st.success("Dataset loaded successfully!")
                st.write("Preview of your dataset:")
                st.dataframe(df)

                temp_file_path = os.path.join(self.output_dir, "temp_data.csv")
                df.to_csv(temp_file_path, index=False)
                df.to_csv("uploaded_dataset.csv", index=False)

                st.write("You can now interact with the chatbot to ask questions about the dataset.")
                self.display_chat_history()
                self.handle_user_input(temp_file_path)
            except Exception as e:
                st.error(f"Error: {e}")

        elif dataset is not None:
            st.success("Previously uploaded dataset loaded successfully!")
            st.write("Preview of your dataset:")
            st.dataframe(dataset)
            temp_file_path = os.path.join(self.output_dir, "temp_data.csv")
            dataset.to_csv(temp_file_path, index=False)

            st.write("You can now interact with the chatbot to ask questions about the dataset.")
            self.display_chat_history()
            self.handle_user_input(temp_file_path)

def main():
    """Main function to initialize the Streamlit app."""
    st.title("Data Analysis Agent")
    st.sidebar.title("Configuration")

    MODEL_OPTIONS = {
        "OpenAI (GPT-4o-mini)": "gpt-4o-mini",
        "OpenAI (GPT-4o)": "gpt-4o",
        "OpenAI (GPT-4)": "gpt-4",
        "OpenAI (GPT-3.5-Turbo)": "gpt-3.5-turbo",
        "Groq (Llama-3.3-70B-Versatile)": "llama-3.3-70b-versatile",
        "Groq (llama3-70b-8192)": "llama3-70b-8192",
    }

    selected_model_name = st.sidebar.selectbox("Select LLM Model", list(MODEL_OPTIONS.keys()))
    selected_model = MODEL_OPTIONS[selected_model_name]
    st.session_state["selected_model"] = selected_model

    is_openai = selected_model.startswith("gpt-")
    is_groq = selected_model.startswith("llama-")

    if is_openai:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        st.session_state["api_key"] = openai_api_key
    elif is_groq:
        groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
        st.session_state["selected_model"] = "groq/" + st.session_state["selected_model"]
        st.session_state["api_key"] = groq_api_key

    if "api_key" in st.session_state and st.session_state["api_key"]:
        agent = create_agent(st.session_state["api_key"], st.session_state["selected_model"])
        app = StreamlitApp(agent=agent)
        app.run()
    else:
        st.sidebar.warning("Please enter the required API Key to use the app.")

if __name__ == "__main__":
    main()
