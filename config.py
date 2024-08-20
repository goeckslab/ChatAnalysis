import streamlit as st
import openai

supported_chat_models = [
    "gpt-4o-mini", "gpt-3.5-turbo", "gpt-3.5-turbo-0125", 
    "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", 
    "gpt-3.5-turbo-16k-0613", "gpt-4", "gpt-4-0125-preview", 
    "gpt-4-1106-preview", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", 
    "gpt-4-turbo-preview", "gpt-4o", "gpt-4o-2024-05-13", 
    "gpt-4o-mini-2024-07-18",
]

def get_openai_key():
    openai_api_key = st.sidebar.text_input(label="Your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.error("OpenAI API key not set for your chat")
        st.stop()
    
    try:
        client = openai.OpenAI(api_key=openai_api_key)
        client.models.list()
    except openai.AuthenticationError as e:
        st.error(e.body["message"])
        st.stop()
    except Exception as e:
        st.error("Not available at this moment. Please try again later.")
        st.stop()

    model = st.sidebar.selectbox(label="Select the model you want", options=supported_chat_models)
    return model, openai_api_key
