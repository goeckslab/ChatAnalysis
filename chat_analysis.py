import streamlit as st
import pandas as pd
import os
import base64
import io
from PIL import Image
from pandasai import Agent
from pandasai.llm import OpenAI
from pandasai.responses.streamlit_response import StreamlitResponse
import config
from helper import detect_image_path

class ChatAnalysisApp:
    def __init__(self):
        self.user_defined_path = os.path.join(os.getcwd(), 'temp')
        self.model, self.openai_api_key = config.get_openai_key()
        self.df = None
        self.agent = None  # Initialize the agent as None
        self.initialize_session_state()

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

    def handle_user_input(self, prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        return prompt

    def process_result(self, result, agent):
        if isinstance(result, str) and "Generated code includes import of" in result and "which is not in whitelist" in result:
            st.code(body=agent.last_code_generated, line_numbers=True)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result, 
                "code_generated": agent.last_code_generated
            })
        elif isinstance(result, str) and os.path.exists(result):
            self.display_image_result(result, agent)
        elif isinstance(result, str) and (image_path := detect_image_path(result)):
            '''
             for edge case: 
             The dataset contains 10 years of data with 12 months recorded. 
             The average monthly values are plotted in 
             '/mount/src/chatanalysis/temp/f0a55a49-b0fa-43d9-8e41-2098421a1544.png'.
            '''
            # image_path = detect_image_path(result)
            self.display_image_text_result(result, image_path, agent)
        else:
            self.display_text_result(result, agent)

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
        os.remove(image_path)

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
        os.remove(result)

    def display_text_result(self, result, agent):
        with st.expander("Executed code"):
            st.code(body=agent.last_code_executed, line_numbers=True)
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result, 
            "code_excuted": agent.last_code_executed
        })
        st.write(result)

    def create_agent(self):
        self.agent = Agent(self.df, config={
            "llm": OpenAI(api_token=self.openai_api_key, model=self.model),
            "save_charts": True,
            "save_charts_path": self.user_defined_path,
        })

    def run(self):
        st.title("ChatAnalysis")

        csv_file = st.file_uploader("Upload your csv file", type=["csv", "tsv"])
        if csv_file is not None:
            self.df = self.load_data(csv_file)
            st.dataframe(self.df.head(5))
            self.create_agent()  # Create the agent only when df is available

        for message in st.session_state.messages:
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
                else:
                    st.write(message["content"])

        if self.agent and (prompt := st.chat_input("What is up?")):  # Check if agent is created
            prompt = self.handle_user_input(prompt)
            with st.chat_message("assistant"):
                result = self.agent.chat(prompt)
                print(result)
                self.process_result(result, self.agent)


if __name__ == "__main__":
    app = ChatAnalysisApp()
    app.run()
            

            
            
        
