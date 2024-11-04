import streamlit as st
import pandas as pd
import os
import base64
import io
from PIL import Image
from pandasai import Agent
from pandasai.llm import OpenAI, BambooLLM, AzureOpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import config
from helper import detect_image_path
from pandasai.exceptions import PandasAIApiCallError
import sys
from st_aggrid import AgGrid
from langchain_groq.chat_models import ChatGroq
import json
from dotenv import load_dotenv
from generate_html_report import generate_html_from_json

st.set_page_config(page_title="Galaxy Chat Analysis", page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'favicon.ico'))

def export_to_galaxy(data, output_path):
    if isinstance(data, pd.DataFrame):
        data.to_csv(output_path, index=False)
        print(f"Exported data to Galaxy at path: {output_path}")
        
        st.toast("The dataset you selected has been saved back to the Galaxy. You can see it on the history bar.")
        
@st.cache_resource
def create_agent(llm_choice, model, df, api_key, user_defined_path):
    llm = None
    if llm_choice == "BambooLLM":
        llm = BambooLLM(api_key=api_key)
    elif llm_choice == "OpenAI":
        llm = OpenAI(api_token=api_key, model=model)
    elif llm_choice == "Groq":
        llm = ChatGroq(api_key=api_key, model_name=model)
    elif llm_choice == "GPT-4o":
        llm = AzureOpenAI(api_token=api_key,
                        azure_endpoint="https://models.inference.ai.azure.com",
                        deployment_name="gpt-4o",
                        api_version="2024-08-01-preview"
                        )
    elif llm_choice == "GPT-4o-mini":
        llm = AzureOpenAI(api_token=api_key,
                        azure_endpoint="https://models.inference.ai.azure.com",
                        deployment_name="gpt-4o-mini",
                        api_version="2024-08-01-preview"
                        )
    elif llm_choice == "Your-Groq-API-Key":
        llm = ChatGroq(api_key=api_key, model_name=model)

    agent = Agent(df, config={
        "llm": llm,
        "save_charts": True,
        "save_charts_path": user_defined_path,
        "custom_whitelisted_dependencies": ["pycaret", "seaborn"],
        "save_logs": False
    })

    return agent

class ChatAnalysisApp:
    '''
a bug in resposeparser:
{'summary': {'type': 'string', 'value': 'The dataset has 10 rows and 13 columns. Columns are: \ufeffYear, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec.'}, 'plot': {'type': 'plot', 'value': '/Users/qjunhao/Documents/GitHub/ChatAnalysis/temp/4ede6325-ae1c-40a3-a098-4675eda2f1ee.png'}}
The dataset has 10 rows and 13 columns. Columns are: Year, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec.
'''
          
    def __init__(self,
                csv_file=None,
                openai_api_key=None,
                bamboollm_key_app=None,
                groq_api_key=None,
                history_html=None,
                free_openai_token=None,
                plots_path=None,
                output_dataset=None,
                groq_api_key_user=None):
        # self.user_defined_path = os.path.join(os.getcwd(), 'temp')
        self.user_defined_path = plots_path
        self.history_html = history_html
        self.llm_choice = None
        self.output_dataset = output_dataset

        # if openai_api_key:
        #     ## to-do: add model choice
        #     self.llm_choice = "OpenAI"
        #     self.openai_api_key = openai_api_key
        #     self.model = "gpt-4o-mini"
        # else:
        
        llm_choice, model, api_key_user = config.configure_llm_options(openai_api_key, groq_api_key_user)
        if llm_choice == "OpenAI":
            self.llm_choice = llm_choice
            self.model = model
            self.api_key = api_key_user
        elif llm_choice == "BambooLLM":
            self.llm_choice = llm_choice
            self.model = "BambooLLM"
            if api_key_user:
                self.api_key = api_key_user
            elif bamboollm_key_app:
                self.api_key = bamboollm_key_app
        elif llm_choice == "Groq":
            self.llm_choice = llm_choice
            self.model = "llama-3.2-90b-text-preview"
            self.api_key = groq_api_key
        elif llm_choice == "GPT-4o":
            self.llm_choice = "GPT-4o"
            self.model = "gpt-4o"
            self.api_key = free_openai_token
        elif llm_choice == "GPT-4o-mini":
            self.llm_choice = "GPT-4o-mini"
            self.model = "gpt-4o-mini"
            self.api_key = free_openai_token
        elif llm_choice == "Your-Groq-API-Key":
            self.llm_choice = llm_choice
            self.model = model
            self.api_key = api_key_user
            
        self.df_loaded = False
        self.df = None
        self.agent = None  # Initialize the agent as None
        self.initialize_session_state()

        if csv_file:
            self.df = self.load_data(csv_file)
            self.df_loaded = True

    @staticmethod
    @st.cache_data
    def load_data(csv_file):
        return pd.read_csv(csv_file, sep=None, engine='python')

    @staticmethod
    def encode_image_to_base64(img_path):
        with open(img_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    @staticmethod
    def decode_base64_to_image(base64_string):
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
            if os.path.exists("chat_history.json"):
                with open("chat_history.json", "r") as json_file:
                    messages = json.load(json_file)
                    for message in messages:
                        if "content_df" in message:
                            message["content"] = pd.DataFrame(message["content_df"])
                    st.session_state.messages = messages
                    
    
    def save_chat_history(self):
        serializable_messages = []

        for message in st.session_state.messages:
            new_message = message.copy()
            if isinstance(message.get("content"), pd.DataFrame):
                # Convert DataFrame to a dictionary  
                new_message["content"] = ""
                new_message["content_df"] = message["content"].to_dict()
            serializable_messages.append(new_message)
        
        with open("chat_history.json", "w") as json_file:
            json.dump(serializable_messages, json_file)
        
        if self.history_html:
            generate_html_from_json("chat_history.json", self.history_html)

    def handle_user_input(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        self.save_chat_history()
    
    def handle_user_input_response(self, prompt):
        self.handle_user_input(prompt)
        with st.chat_message("assistant"):
            result = None
            try:
                with st.spinner('Generating response...'):
                    result = self.agent.chat(prompt)
            except PandasAIApiCallError as e:
                st.write(f"The BambooLLM free tier has a limit of 100 API calls per month. \
                        We have reached the limit. \
                        Please use the OpenAI model to continue the analysis.")
            self.process_result(result, self.agent)

    def process_result(self, result, agent):
        if isinstance(result, pd.DataFrame):
            self.dispaly_dataframe_result(result, agent)
        elif isinstance(result, str) and "Generated code includes import of" in result and "which is not in whitelist" in result:
            st.code(body=agent.last_code_generated, line_numbers=True)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result, 
                "code_generated": agent.last_code_generated
            })
        elif isinstance(result, str) and os.path.exists(result):
            # print("only image")
            self.display_image_result(result, agent)
        elif isinstance(result, str) and (image_path := detect_image_path(result)):
            # image_path = detect_image_path(result)
            # print("image and text")
            if "No such file or directory" in result:
                st.write(result)
            else:
                self.display_image_text_result(result, image_path, agent)
        else:
            self.display_text_result(result, agent)
        self.save_chat_history()

    def dispaly_dataframe_result(self, result, agent):
        with st.expander("Executed code"):
            st.code(body=agent.last_code_executed, line_numbers=True)
        AgGrid(result, height=220, key=f'export_df_{len(st.session_state.messages)}', enable_enterprise_modules=False)
        st.session_state.messages.append({
            "role": "assistant",
            "content": result,
            "code_excuted": agent.last_code_executed
        })
        if st.button(f'Export this dataset back to Galaxy', key=f'export_{len(st.session_state.messages)-1}'):
            print("processing button")
            if self.output_dataset:
                export_to_galaxy(result, self.output_dataset)           

    def display_image_text_result(self, result, image_path, agent):
        with st.expander("Executed code"):
            st.code(body=agent.last_code_executed, line_numbers=True)
        image = self.encode_image_to_base64(image_path)
        st.image(self.decode_base64_to_image(image))
        st.session_state.messages.append({
            "role": "assistant", 
            "image": image, 
            "content": result,
            "code_excuted": agent.last_code_executed
        })

    def display_image_result(self, result, agent):
        with st.expander("Executed code"):
            st.code(body=agent.last_code_executed, line_numbers=True)
        image = self.encode_image_to_base64(result)
        st.image(self.decode_base64_to_image(image))
        st.session_state.messages.append({
            "role": "assistant", 
            "image": image, 
            "code_excuted": agent.last_code_executed
        })

    def display_text_result(self, result, agent):
        with st.expander("Executed code"):
            st.code(body=agent.last_code_executed, line_numbers=True)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result, 
            "code_excuted": agent.last_code_executed
        })
        st.write(result)

    def run(self):
        st.title("Galaxy Chat Analysis")
        seleted_question = config.display_example_questions()
        config.display_notes()
        
        if not self.df_loaded:
            csv_file = st.file_uploader("Upload your csv file", type=["csv", "tsv"])
            if csv_file is not None:
                self.df = self.load_data(csv_file)
                self.df_loaded = True
        
        if self.df_loaded:
            AgGrid(self.df.head(5), height=220, enable_enterprise_modules=False)
            self.agent = create_agent(self.llm_choice, self.model, self.df, self.api_key, self.user_defined_path)

            for idx, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    if "image" in message and "content" in message:
                        with st.expander("Executed code"):
                            st.code(body=message["code_excuted"], line_numbers=True)
                        st.image(self.decode_base64_to_image(message["image"]))
                        st.write(message["content"])
                    elif "image" in message:
                        with st.expander("Executed code"):
                            st.code(body=message["code_excuted"], line_numbers=True)
                        st.image(self.decode_base64_to_image(message["image"]))
                    elif "code_generated" in message:
                        st.code(body=message["code_generated"], line_numbers=True)
                    elif isinstance(message['content'], pd.DataFrame):
                        with st.expander("Executed code"):
                            st.code(body=message["code_excuted"], line_numbers=True)
                        df = message["content"]
                        print(f"lookat the df: {df}")
                        AgGrid(df, height=220, enable_enterprise_modules=False, key=f'export_df_{idx}')
                        # print(message)
                        if st.button(f'Export this dataset back to Galaxy', key=f'export_{idx}'):
                            if self.output_dataset:
                                df_to_export = pd.DataFrame(message["content"].to_dict())
                                export_to_galaxy(df_to_export, self.output_dataset)
                    elif message["role"] == "assistant":
                        with st.expander("Executed code"):
                            st.code(body=message["code_excuted"], line_numbers=True)
                        st.write(message["content"])
                    else:
                        st.markdown(message["content"])

            if seleted_question and self.agent:
                self.handle_user_input_response(seleted_question)

            if self.agent and (prompt := st.chat_input("Ask a question about your data (You can click on an example question in the sidebar).")):
                self.handle_user_input_response(prompt)


if __name__ == "__main__":
    #not sure in docker, at start, the csv_file in sys.argv[4], the openai_api_key in sys.argv[5]
    # the argv was = ['something unknown', 'streamlit', 'run', 'chat_analysis.py', '*.csv', 'sk-xxxx']
    # bamboollm_key_app_file = sys.argv[1] if len(sys.argv) > 1 else None
    # groq_api_key_file = sys.argv[2] if len(sys.argv) > 2 else None

    openai_api_key_file = sys.argv[1] if len(sys.argv) > 1 else None
    groq_api_key_user = sys.argv[2] if len(sys.argv) > 2 else None
    chat_history_html = sys.argv[3] if len(sys.argv) > 3 else None
    output_dataset = sys.argv[4] if len(sys.argv) > 4 else None
    plots_path = sys.argv[5] if len(sys.argv) > 5 else None
    csv_file = sys.argv[6] if len(sys.argv) > 6 else None

    
    openai_api_key = None
    groq_api_key_user = None
    bamboollm_key_app = None
    groq_api_key = None
    free_openai_token = None
    if openai_api_key_file:
        with open(openai_api_key_file, 'r') as f:
            openai_api_key = f.read().strip()
    
    if groq_api_key_user:
        with open(groq_api_key_user, 'r') as f:
            groq_api_key_user = f.read().strip()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    env_path = os.path.join(script_dir, '.env')     
    if os.path.exists(env_path):
        print(".env loaded")
        load_dotenv(dotenv_path=env_path)
        if os.getenv("GROQ_API_KEY"):
            groq_api_key = os.getenv("GROQ_API_KEY")
        if os.getenv("BAMBOOLLM_API_KEY"):
            bamboollm_key_app = os.getenv("BAMBOOLLM_API_KEY")
        if os.getenv("GITHUB_ACCESS_TOKEN"):
            free_openai_token = os.getenv("GITHUB_ACCESS_TOKEN")
    # print(groq_api_key)
    app = ChatAnalysisApp(csv_file,
                        openai_api_key,
                        bamboollm_key_app,
                        groq_api_key,
                        chat_history_html,
                        free_openai_token,
                        plots_path,
                        output_dataset,
                        groq_api_key_user
                        )
    app.run()
            
  
            
        
