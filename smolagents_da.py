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

# Set logging level to DEBUG for detailed logs
# logging.basicConfig(level=logging.DEBUG)

load_dotenv()

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
            "pandas", "numpy", "matplotlib", "seaborn", "scipy", "statsmodels",
            "sklearn", "pycaret", "plotly", "joblib", "io"
        ],
        max_steps=20,
    )

# def clean_text(text):
#     lines = text.splitlines()
#     cleaned_lines = []
#     for line in lines:
#         tokens = line.split()
#         if tokens and all(len(token) == 1 for token in tokens):
#             cleaned_lines.append("".join(tokens))
#         else:
#             cleaned_lines.append(line)
#     return "\n".join(cleaned_lines)

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


class StreamlitApp:
    def __init__(self, agent):
        self.agent = agent
        self.output_dir = "outputs_smolagents"
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

    # --- Modified: Save chat history including memory ---
    def save_chat_history(self):
        history = {
            "messages": st.session_state.get("messages", []),
            "eda_report": st.session_state.get("eda_report", ""),
            "memory": list(st.session_state.get("memory", []))
        }
        with open("chat_history.json", "w") as f:
            json.dump(history, f)

    # --- Modified: Load chat history including memory ---
    def load_chat_history(self):
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as f:
                history = json.load(f)
                st.session_state["messages"] = history.get("messages", [])
                st.session_state["eda_report"] = history.get("eda_report", "")
                memory_list = history.get("memory", [])
                st.session_state["memory"] = deque(memory_list, maxlen=15)

    def load_dataset_preview(self):
        if os.path.exists("uploaded_dataset.csv"):
            return pd.read_csv("uploaded_dataset.csv")
        return None

    def format_memory_steps(self):
        middle_steps = ""
        try:
            full_steps = self.agent.memory.get_full_steps()
            for idx, step in enumerate(full_steps, start=1):
                has_content = False
                for key in ["system_prompt", "tool_calls", "observations", "model_output", "action_output", "error"]:
                    if key in step and step[key]:
                        if isinstance(step[key], str) and step[key].strip():
                            has_content = True
                        elif not isinstance(step[key], str):
                            has_content = True
                if not has_content:
                    continue
                step_number = step.get("step") or idx
                middle_steps += f"**Step {step_number}:**\n"
                if step.get("system_prompt"):
                    middle_steps += f"System Prompt: {step['system_prompt']}\n"
                if step.get("tool_calls"):
                    tool_names = []
                    for tc in step["tool_calls"]:
                        if isinstance(tc, dict) and "function" in tc and "name" in tc["function"]:
                            tool_names.append(tc["function"]["name"])
                    if tool_names:
                        middle_steps += f"Tool Call(s): {', '.join(tool_names)}\n"
                if step.get("observations"):
                    middle_steps += f"Observations: {step['observations']}\n"
                if step.get("model_output"):
                    middle_steps += f"Model Output: {step['model_output']}\n"
                if step.get("action_output"):
                    middle_steps += f"Action Output: {step['action_output']}\n"
                if step.get("error"):
                    middle_steps += f"Error: {step['error']}\n"
                middle_steps += "\n"
        except Exception as e:
            logging.error("Error retrieving memory steps: %s", e)
        return middle_steps

    def display_response(self, explanation, plot_paths, file_paths, next_steps_suggestion, middle_steps="", candidate_solutions=None):
        with st.chat_message("assistant"):
            # Clean explanation and next steps text.
            explanation = clean_text(explanation)
            next_steps_suggestion = clean_text(next_steps_suggestion)
            
            # If candidate solutions are provided, display them separately.
            if candidate_solutions is not None:
                st.markdown("### Candidate Solutions")
                for idx, candidate in enumerate(candidate_solutions, start=1):
                    with st.expander(f"Candidate {idx}: {candidate.get('option', 'Option')}"):
                        st.markdown(f"**Explanation:** {candidate.get('explanation', '')}")
                        st.markdown(f"**Pros:** {candidate.get('pros', '')}")
                        st.markdown(f"**Cons:** {candidate.get('cons', '')}")
                        # A button to allow the user to refine this candidate solution.
                        if st.button("Refine this solution", key=f"refine_candidate_{idx}"):
                            # Pre-fill input with candidate details for refinement.
                            st.session_state["prefilled_input"] = candidate.get("option", "") + " " + candidate.get("explanation", "")
            else:
                # Display the explanation text normally.
                if "count" in explanation and "mean" in explanation and "std" in explanation:
                    st.code(explanation)
                else:
                    st.markdown(explanation)
            
            # Display intermediate steps if available.
            if middle_steps:
                with st.expander("View Intermediate Steps"):
                    st.markdown(middle_steps)
            
            # Display any generated plots.
            for plot_path in plot_paths:
                if plot_path and os.path.exists(plot_path):
                    image = Image.open(plot_path)
                    st.image(image, caption="Generated Plot")
            
            # Display file download buttons for any generated files.
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
            
            # Display clickable next steps suggestions.
            if not candidate_solutions and next_steps_suggestion:
                suggestions = [s.strip() for s in next_steps_suggestion.split("\n") if s.strip()]
                self.display_suggestion_buttons(suggestions)
                st.markdown("Please let me know if you want to proceed with any of the suggestions or ask any other questions.")

    # def display_response(self, explanation, plot_paths, file_paths, next_steps_suggestion, middle_steps=""):
    #     with st.chat_message("assistant"):
    #         explanation = clean_text(explanation)
    #         next_steps_suggestion = clean_text(next_steps_suggestion)
    #         # Display explanation
    #         if "count" in explanation and "mean" in explanation and "std" in explanation:
    #             st.code(explanation)
    #         else:
    #             st.markdown(explanation)
    #         # Display intermediate steps in an expander if available
    #         if middle_steps:
    #             with st.expander("View Intermediate Steps"):
    #                 st.markdown(middle_steps)
    #         # Display plots if any
    #         for plot_path in plot_paths:
    #             if plot_path and os.path.exists(plot_path):
    #                 image = Image.open(plot_path)
    #                 st.image(image, caption="Generated Plot")
    #         # Display file download buttons if any
    #         for file_path in file_paths:
    #             if file_path and os.path.exists(file_path):
    #                 unique_key = str(uuid.uuid4())
    #                 with open(file_path, "rb") as f:
    #                     st.download_button(
    #                         label=f"Download {os.path.basename(file_path)}",
    #                         data=f,
    #                         file_name=os.path.basename(file_path),
    #                         key=f"download_{unique_key}"
    #                     )
    #         # Convert next steps suggestions into clickable buttons
    #         if next_steps_suggestion:
    #             suggestions = [s.strip() for s in next_steps_suggestion.split("\n") if s.strip()]
    #             self.display_suggestion_buttons(suggestions)
    #             # for i, suggestion in enumerate(suggestions):
    #             #     if st.button(suggestion, key=suggestion.replace(' ', '_')):
    #             #         temp_file_path = st.session_state.get("temp_file_path", "")
    #             #         self.handle_user_input(temp_file_path, suggestion)
    #             st.markdown("Please let me know if you want to proceed with any of the suggestions or ask any other questions.")
    #         # if next_steps_suggestion:
    #         #     suggestion_container = st.empty()
    #         #     with suggestion_container.container():
    #         #         st.markdown("**Next Steps Suggestion:**")
    #         #         suggestions = [s.strip() for s in next_steps_suggestion.split("\n") if s.strip()]
    #         #         for suggestion in suggestions:
    #         #             # If a button is clicked, clear the container immediately
    #         #             if st.button(suggestion, key=f"btn_{suggestion.replace(' ', '_')}"):
    #         #                 suggestion_container.empty()  # Remove all suggestion buttons
    #         #                 temp_file_path = st.session_state.get("temp_file_path", "")
    #         #                 self.handle_user_input(temp_file_path, suggestion)
    #         #                 # Optionally, break here to stop processing further buttons
    #         #                 break
    #         #     st.markdown("Please let me know if you want to proceed with any of the suggestions or ask any other questions.")

    # def display_chat_history(self):
    #     messages = st.session_state["messages"]
    #     for idx, message in enumerate(messages):
    #         # Render the chat message content in its own chat block.
    #         with st.chat_message(message["role"]):
    #             if "count" in message["content"] and "mean" in message["content"] and "std" in message["content"]:
    #                 st.code(message["content"])
    #             else:
    #                 st.markdown(message["content"])
    #             if "middle_steps" in message and message["middle_steps"]:
    #                 with st.expander("View Intermediate Steps"):
    #                     st.markdown(message["middle_steps"])
    #             if "image_paths" in message:
    #                 for plot_path in message["image_paths"]:
    #                     if os.path.exists(plot_path):
    #                         image = Image.open(plot_path)
    #                         st.image(image, caption="Generated Plot")
    #             if "file_paths" in message:
    #                 for file_path in message["file_paths"]:
    #                     if os.path.exists(file_path):
    #                         unique_key = str(uuid.uuid4())
    #                         with open(file_path, "rb") as f:
    #                             st.download_button(
    #                                 label=f"Download {os.path.basename(file_path)}",
    #                                 data=f,
    #                                 file_name=os.path.basename(file_path),
    #                                 key=f"history_download_{unique_key}"
    #                             )
    #         # For suggestions, only render clickable buttons for the last message.
    #         if "next_steps_suggestion" in message and message["next_steps_suggestion"]:
    #             if idx == len(messages) - 1:
    #                 st.markdown("**Next Steps Suggestion:**")
    #                 suggestions = [s.strip() for s in message["next_steps_suggestion"].split("\n") if s.strip()]
    #                 for suggestion in suggestions:
    #                     if st.button(suggestion, key=f"last_{suggestion.replace(' ', '_')}"):
    #                         temp_file_path = st.session_state.get("temp_file_path", "")
    #                         self.handle_user_input(temp_file_path, suggestion)
    #             else:
    #                 # For previous messages, render suggestions as plain text.
    #                 st.markdown(f"**Next Steps Suggestion:** {message['next_steps_suggestion']}")

    # def display_chat_history(self):
    #     messages = st.session_state["messages"]
    #     for idx, message in enumerate(messages):
    #         with st.chat_message(message["role"]):
    #             if "count" in message["content"] and "mean" in message["content"] and "std" in message["content"]:
    #                 st.code(message["content"])
    #             else:
    #                 st.markdown(message["content"])
    #             if "middle_steps" in message and message["middle_steps"]:
    #                 with st.expander("View Intermediate Steps"):
    #                     st.markdown(message["middle_steps"])
    #             if "image_paths" in message:
    #                 for plot_path in message["image_paths"]:
    #                     if os.path.exists(plot_path):
    #                         image = Image.open(plot_path)
    #                         st.image(image, caption="Generated Plot")
    #             if "file_paths" in message:
    #                 for file_path in message["file_paths"]:
    #                     if os.path.exists(file_path):
    #                         unique_key = str(uuid.uuid4())
    #                         with open(file_path, "rb") as f:
    #                             st.download_button(
    #                                 label=f"Download {os.path.basename(file_path)}",
    #                                 data=f,
    #                                 file_name=os.path.basename(file_path),
    #                                 key=f"history_download_{unique_key}"
    #                             )
    #             if "next_steps_suggestion" in message and message["next_steps_suggestion"] and idx != len(messages) - 1:
    #                 st.markdown(f"**Next Steps Suggestion:** \n* {message['next_steps_suggestion']}")
    #             # elif "next_steps_suggestion" in message and idx == len(messages) - 1:
    #             #     suggestions = [s.strip() for s in message["next_steps_suggestion"].split("\n") if s.strip()]
    #             #     self.display_suggestion_buttons(suggestions)
                    

                        
        # Render suggestion buttons in a dedicated container after the chat history
        # if messages:
        #     last_message = messages[-1]
        #     # Only display suggestion buttons if the last message is from the assistant and has suggestions
        #     if last_message["role"] == "assistant" and last_message.get("next_steps_suggestion"):
        #         suggestions = [s.strip() for s in last_message["next_steps_suggestion"].split("\n") if s.strip()]
        #         self.display_suggestion_buttons(suggestions)

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
                            st.image(image, caption="Generated Plot")

                # Display file download buttons for any generated files.
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

    def get_agent_prompt(self, temp_file_path, user_question):
        memory_history = ""
        if st.session_state.get("memory"):
            memory_history = "\n".join(st.session_state["memory"])
        if "exploratory data analysis" in user_question.lower():
            return (
                "You are an expert assistant who can solve any task using code blobs." 
                "To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences."
                f"The dataset is saved at {temp_file_path}. {user_question}\n\n"
                "- Always suggest possible next steps for data analysis at the end of the answer, unless the user is explicitly asking for suggestions.\n"
                "- If a plot or file is generated, save it in the outputs/ directory with a random numerical suffix to prevent overwrites.\n"
                "- Do not generate filenames like 'random_forest_model_XXXX.joblib'.\n"
                "- Always call the final_answer tool, providing the final answer in the following dictionary format (do not format as a JSON code block):\n"
                '{ "explanation": ["Your explanation here, in plain text. This can include detailed information or step-by-step guidance."], '
                '"plots": ["<path_to_the_image>" (leave empty if no plots are needed)], '
                '"files": ["<path_to_the_file>" (leave empty if no files are needed)], '
                '"next_steps_suggestion": ["List of possible next questions the user could ask to gain further insights. They should be questions. Only include this when the user has not explicitly asked for suggestions."] }'
            )
        elif "Summarize the previous conversation in a concise manner." in user_question.lower():
            return (
                "You are an expert assistant who can solve any task using code blobs." 
                "To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences."
                f"Previous conversation:\n{memory_history}\n\nCurrent Question:{user_question}\n\n"
                "- Always suggest possible next steps for data analysis at the end of the answer, unless the user is explicitly asking for suggestions.\n"
                "- If a plot or file is generated, save it in the outputs/ directory with a random numerical suffix to prevent overwrites.\n"
                "- Do not generate filenames like 'random_forest_model_XXXX.joblib'.\n"
                "- Always call the final_answer tool, providing the final answer in the following dictionary format (do not format as a JSON code block):\n"
                '{ "explanation": ["Your explanation here, in plain text. This can include detailed information or step-by-step guidance."], '
                '"plots": ["<path_to_the_image>" (leave empty if no plots are needed)], '
                '"files": ["<path_to_the_file>" (leave empty if no files are needed)], '
                '"next_steps_suggestion": ["List of possible next questions the user could ask to gain further insights. They should be questions. Only include this when the user has not explicitly asked for suggestions."] }'
            )
        else:
            return (
                f"Previous conversation:\n{memory_history}\n\n"
                "You are an expert assistant who can solve any task using code blobs." 
                "To solve the task, you must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences."
                f"The dataset is saved at {temp_file_path}. Current Question: {user_question}\n\n"
                "- Before answering, please analyze the user's question. If you determine the question is multifaceted, ambiguous, or covers several aspects, provide three distinct candidate solutions. For each candidate, include:\n"
                "   - An 'option' title,\n"
                "   - A detailed 'explanation',\n"
                "   - A list of 'pros',\n"
                "   - A list of 'cons'.\n"
                "- If the question is straightforward, provide a single concise answer following the standard format. But most questions should be strightforward.\n"
                "- Always include next step suggestions at the end.\n"
                "- If a plot or file is generated, save it in the outputs/ directory with a random numerical suffix to prevent overwrites.\n"
                "- Do not generate filenames like 'random_forest_model_XXXX.joblib'.\n"
                "- Always call the final_answer tool, providing the final answer in one of the following dictionary formats (do not format as a JSON code block):\n\n"
                "Simple answer format:\n"
                '{ "explanation": ["Your explanation text. in plain text. This can include detailed information or step-by-step guidance."], "plots": ["<path_to_image>"], "files": ["<path_to_file>"], "next_steps_suggestion": ["Suggestion 1", "Suggestion 2"] }\n\n'
                "Multiple candidate solutions format:\n"
                '{ "candidate_solutions": [ { "option": "Solution 1", "explanation": "Detailed explanation...", "pros": "Pros...", "cons": "Cons..." }, { "option": "Solution 2", "explanation": "Detailed explanation...", "pros": "Pros...", "cons": "Cons..." }, { "option": "Solution 3", "explanation": "Detailed explanation...", "pros": "Pros...", "cons": "Cons..." } ], "next_steps_suggestion": ["Which option would you like to refine?", "Or ask for more details on a candidate solution."] }'
            )

    def handle_user_input(self, temp_file_path, user_question):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state["messages"].append({"role": "user", "content": user_question})
        st.session_state["memory"].append(f"User: {user_question}")
        self.save_chat_history()
        # self.display_chat_history()
        with st.spinner("Thinking..."):
            prompt = self.get_agent_prompt(temp_file_path, user_question)
            response = self.agent.run(prompt)
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
        with st.spinner("Running EDA..."):
            try:
                eda_response = self.agent.run(self.get_agent_prompt(temp_file_path, eda_query))
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

                st.success("EDA complete! Download the report below:")
                st.download_button(
                    label="Download EDA Report",
                    data=html_content,
                    file_name="eda_report.html",
                    mime="text/html"
                )

                eda_result_message = {
                    "role": "assistant",
                    "content": report_text,
                    "image_paths": parsed.get("plots", []) if parsed else [],
                    "file_paths": parsed.get("files", []) if parsed else [],
                    "next_steps_suggestion": "  \n* ".join(parsed.get("next_steps_suggestion", [])) if parsed else "",
                    "middle_steps": middle_steps
                }
                st.session_state["messages"].append(eda_result_message)
                st.session_state["memory"].append(f"Assistant (EDA): {report_text}")

                self.display_response(
                    explanation=report_text,
                    plot_paths=parsed.get("plots", []) if parsed else [],
                    file_paths=parsed.get("files", []) if parsed else [],
                    next_steps_suggestion="  \n* ".join(parsed.get("next_steps_suggestion", [])) if parsed else "",
                    middle_steps=middle_steps
                )
                self.save_chat_history()
            except Exception as e:
                st.error(f"Error during EDA: {e}")

    def summarize_chat_history(self):
        summary_prompt = "Summarize the previous conversation in a concise manner.\n"
        self.handle_user_input("", summary_prompt)

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

    # def process_response(self, response, middle_steps=""):
    #     if isinstance(response, dict):
    #         message = {
    #             "explanation": "\n".join(response.get("explanation", [])),
    #             "plots": response.get("plots", []),
    #             "files": response.get("files", []),
    #             "next_steps_suggestion": "  \n* ".join(response.get("next_steps_suggestion", [])),
    #             "middle_steps": middle_steps
    #         }
    #         if not message["plots"] and not message["files"]:
    #             message["explanation"] += "\nLLM did not generate any plots or files."
    #         self.display_response(
    #             message["explanation"],
    #             message["plots"],
    #             message["files"],
    #             message["next_steps_suggestion"],
    #             message["middle_steps"]
    #         )
    #         st.session_state["messages"].append({
    #             "role": "assistant",
    #             "content": message["explanation"],
    #             "image_paths": message["plots"],
    #             "file_paths": message["files"],
    #             "next_steps_suggestion": message["next_steps_suggestion"],
    #             "middle_steps": message["middle_steps"]
    #         })
    #         st.session_state["memory"].append(f"Assistant: {message['explanation']}")
    #     elif isinstance(response, str):
    #         parsed_message = self.parse_response_content(response)
    #         if parsed_message:
    #             self.process_response(parsed_message, middle_steps)
    #         else:
    #             st.session_state["messages"].append({
    #                 "role": "assistant",
    #                 "content": f"Response received:\n```json\n{response}\n```"
    #             })
    #     elif hasattr(response, 'role') and hasattr(response, 'content'):
    #         role = getattr(response, 'role', 'assistant')
    #         content = getattr(response, 'content', '')
    #         parsed_message = self.parse_response_content(content)
    #         if parsed_message:
    #             message = parsed_message
    #             if not message["plots"] and not message["files"]:
    #                 message["explanation"] += "\nLLM did not generate any plots or files."
    #         else:
    #             message = {
    #                 "explanation": content + "\nLLM did not generate any plots or files.",
    #                 "plots": [],
    #                 "files": [],
    #                 "next_steps_suggestion": "",
    #                 "middle_steps": ""
    #             }
    #         self.display_response(
    #             message["explanation"],
    #             message["plots"],
    #             message["files"],
    #             message["next_steps_suggestion"],
    #             middle_steps
    #         )
    #         st.session_state["messages"].append({
    #             "role": role,
    #             "content": message["explanation"],
    #             "image_paths": message["plots"],
    #             "file_paths": message["files"],
    #             "next_steps_suggestion": message["next_steps_suggestion"],
    #             "middle_steps": middle_steps
    #         })
    #         st.session_state["memory"].append(f"{role.capitalize()}: {message['explanation']}")
    #     else:
    #         st.session_state["messages"].append({
    #             "role": "assistant",
    #             "content": f"Response received:\n```json\n{response}\n```"
    #         })


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
                    # Display candidate solutions
                    self.display_response(
                        explanation="Multiple candidate solutions generated.",
                        plot_paths=[],
                        file_paths=[],
                        next_steps_suggestion=next_steps,
                        middle_steps=middle_steps,
                        candidate_solutions=candidate_list
                    )
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
                    self.display_response(
                        message["explanation"],
                        message["plots"],
                        message["files"],
                        message["next_steps_suggestion"],
                        message["middle_steps"]
                    )
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
                self.display_response(
                    explanation="",
                    plot_paths=[],
                    file_paths=[],
                    next_steps_suggestion=next_steps,
                    middle_steps=middle_steps,
                    candidate_solutions=candidate_list
                )
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
                self.display_response(
                    message["explanation"],
                    message["plots"],
                    message["files"],
                    message["next_steps_suggestion"],
                    message["middle_steps"]
                )
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

        # Load existing EDA report if it exists.
        # eda_path = os.path.join(self.output_dir, "eda_report.html")
        # if os.path.exists(eda_path):
        #     st.session_state["eda_report"] = eda_path
        
        # Determine which dataset to use.
        uploaded_file = st.file_uploader("Upload your dataset (CSV or TSV)", type=["csv", "tsv"])
        df = None
        if uploaded_file is not None:
            try:
                df = self.load_dataset(uploaded_file)
                st.session_state["eda_report"] = None
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
        elif self.load_dataset_preview() is not None:
            df = self.load_dataset_preview()

        if df is not None:
            st.success("Dataset loaded successfully!")
            st.write("Preview of your dataset:")
            st.dataframe(df)
            self.display_chat_history()

            temp_file_path = os.path.join(self.output_dir, "temp_data.csv")
            st.session_state["temp_file_path"] = temp_file_path
            df.to_csv(temp_file_path, index=False)
            if uploaded_file is not None:
                df.to_csv("uploaded_dataset.csv", index=False)
            
            user_question = st.chat_input("Ask a question about the dataset")
            if user_question or st.session_state.get("prefilled_input"):
                user_question = st.session_state.get("prefilled_input", user_question)
                st.session_state["prefilled_input"] = None
                self.handle_user_input(temp_file_path, user_question)

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
                self.run_eda(temp_file_path)
            elif not self.has_eda_history():
                self.run_eda(temp_file_path)

            # st.write("You can now interact with the chatbot to ask questions about the dataset.")
            
            if os.path.exists(temp_file_path):
                if st.sidebar.button("Summary Statistics", key="summary_stats"):
                    self.handle_user_input(temp_file_path, "What are the summary statistics for the dataset?")
                if st.sidebar.button("Missing Values", key="missing_values"):
                    self.handle_user_input(temp_file_path, "What are the missing values in the dataset?")
                if st.sidebar.button("Correlation Matrix", key="corr_matrix"):
                    self.handle_user_input(temp_file_path, "Show the correlation matrix of the features.")
                if st.sidebar.button("Numerical Feature Distribution", key="num_dist"):
                    self.handle_user_input(temp_file_path, "Show the distribution of numerical features.")

            st.sidebar.markdown("---")
            st.sidebar.markdown("### Summarize Chat History")
            if st.sidebar.button("Summarize Chat", key="summarize_chat"):
                self.summarize_chat_history()
            
        else:
            st.info("Please upload a dataset.")

def main():
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
    is_openai = selected_model.startswith("gpt-") or selected_model == "gpt-4o"
    is_groq = selected_model.startswith("llama-")
    if is_openai:
        openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
        st.session_state["api_key"] = openai_api_key
    elif is_groq:
        groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
        st.session_state["selected_model"] = "groq/" + st.session_state["selected_model"]
        st.session_state["api_key"] = groq_api_key

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


    if "api_key" in st.session_state and st.session_state["api_key"]:
        agent = create_agent(st.session_state["api_key"], st.session_state["selected_model"])
        app = StreamlitApp(agent=agent)
        app.run()
    else:
        st.sidebar.warning("Please enter the required API Key to use the app.")

if __name__ == "__main__":
    main()
