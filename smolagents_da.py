import os
import pandas as pd
from PIL import Image
from collections import deque
from dotenv import load_dotenv
import streamlit as st
from smolagents import CodeAgent, LiteLLMModel
import json
import uuid

load_dotenv()


@st.cache_resource
def create_agent(api_key, model_id):
    model = LiteLLMModel(model_id=model_id, api_key=api_key)
    return MemoryCodeAgent(
        tools=[],
        model=model,
        additional_authorized_imports=["pandas", "numpy", "matplotlib", "PIL", "seaborn", "sklearn", "pycaret", "plotly", "joblib", "io"],
    )

class MemoryCodeAgent(CodeAgent):
    chat_history = deque(maxlen=10)

    def __init__(self, tools, model, **kwargs):
        super().__init__(tools=tools, model=model, **kwargs)
        

    def run(self, prompt):
        current_history = f"User: {prompt}"
        history_prompt = "Previous conversation:\n" + "\n".join(self.chat_history)
        task = history_prompt + "\ncurrent question: " + prompt
        response = super().run(task)
        current_history += f"\nAgent: {response}"
        MemoryCodeAgent.chat_history.append(current_history)
        return response

class StreamlitApp:
    def __init__(self, agent):
        self.agent = agent
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)

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

    def process_response(self, response):
        if isinstance(response, dict):
            explanation = "\n".join(response.get("explanation", []))
            plot_paths = response.get("plots", [])
            file_paths = response.get("files", [])
            next_steps_suggestion = "  \n* ".join(response.get("next_steps_suggestion", []))
            self.display_response(explanation, plot_paths, file_paths, next_steps_suggestion)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": explanation,
                "image_paths": plot_paths,
                "file_paths": file_paths,
                "next_steps_suggestion": next_steps_suggestion,
            })
        elif isinstance(response, str):
            try:
                response_dict = json.loads(response)
                self.process_response(response_dict)
            except json.JSONDecodeError:
                st.session_state["messages"].append({"role": "assistant", "content": f"Response received:\n```json\n{response}\n```"})
        elif hasattr(response, 'role') and hasattr(response, 'content'):
            # Assuming LiteLLM message type
            # st.write("here")
            # st.write(response)
            role = getattr(response, 'role', 'assistant')
            content = getattr(response, 'content', '')
            try:
                parsed_content = json.loads(content)  # Parse the content if it is a JSON string
                explanation = "\n".join(parsed_content.get("explanation", []))
                plot_paths = parsed_content.get("plots", [])
                file_paths = parsed_content.get("files", [])
                next_steps_suggestion = "  \n* ".join(parsed_content.get("next_steps_suggestion", []))
                if not plot_paths and not file_paths:
                    explanation += "\n" + "LLM did not generate any plots or files."
                # st.write(len(plot_paths))
                # st.write("------------------------------------")
                # st.write(len(file_paths))
            except json.JSONDecodeError:
                st.write("here2")
                explanation = content
                explanation += "\n" + "LLM did not generate any plots or files."
                plot_paths = []
                file_paths = []
                next_steps_suggestion = ""

            self.display_response(explanation, plot_paths, file_paths, next_steps_suggestion)

            st.session_state["messages"].append({
                "role": role,
                "content": explanation,
                "image_paths": plot_paths,
                "file_paths": file_paths,
                "next_steps_suggestion": next_steps_suggestion
            })
        else:
            st.session_state["messages"].append({"role": "assistant", "content": f"Response received:\n```json\n{response}\n```"})


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
                            st.write(f"**Next Steps Suggestion:**  \n* {message['next_steps_suggestion']}")

                if user_question := st.chat_input("Ask a question about the dataset"):
                    with st.chat_message("user"):
                        st.markdown(user_question)

                    st.session_state["messages"].append({"role": "user", "content": user_question})

                    with st.spinner("Generating response..."):
                        response = self.agent.run(
                            f"""
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
                        )


                    self.process_response(response)
                    self.save_chat_history()

            except Exception as e:
                st.error(f"Error: {e}")

        elif dataset is not None:
            st.success("Previously uploaded dataset loaded successfully!")
            st.write("Preview of your dataset:")
            st.dataframe(dataset)
            temp_file_path = os.path.join(self.output_dir, "temp_data.csv")
            dataset.to_csv(temp_file_path, index=False)

            st.write("You can now interact with the chatbot to ask questions about the dataset.")

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
                            st.write(f"**Next Steps Suggestion:**  \n* {message['next_steps_suggestion']}")


            if user_question := st.chat_input("Ask a question about the dataset"):
                with st.chat_message("user"):
                    st.markdown(user_question)

                st.session_state["messages"].append({"role": "user", "content": user_question})

                with st.spinner("Generating response..."):
                    response = self.agent.run(
                            f"""
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
                        )

                self.process_response(response)
                self.save_chat_history()

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