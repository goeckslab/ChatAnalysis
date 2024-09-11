import streamlit as st
import openai

supported_chat_models = [
    "gpt-4o-mini", 
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125", 
    "gpt-3.5-turbo-1106", 
    "gpt-3.5-turbo-0613", 
    "gpt-3.5-turbo-16k", 
    "gpt-3.5-turbo-16k-0613", 
    "gpt-4", 
    "gpt-4-0125-preview", 
    "gpt-4-1106-preview", 
    "gpt-4-0613", 
    "gpt-4-32k", 
    "gpt-4-32k-0613", 
    "gpt-4-turbo-preview", 
    "gpt-4o", 
    "gpt-4o-2024-05-13", 
    "gpt-4o-mini-2024-07-18",
]

def get_openai_key(openai_api_key):
    if not openai_api_key:
        openai_api_key = st.sidebar.text_input(label="Your OpenAI API Key:", type="password")
    if not openai_api_key:
        st.error("OpenAI API key not set for your chat")
        st.stop()
    
    # try:
    #     client = openai.OpenAI(api_key=openai_api_key)
    #     client.models.list()
    # except Exception as e:
    #     st.error(e)
    #     st.stop()

    model = st.sidebar.selectbox(label="Select the model you want", options=supported_chat_models)
    return model, openai_api_key

def configure_llm_options(openai_api_key):
    available_options = ["GPT-4o", "Groq", "BambooLLM", "OpenAI", "Your BambooLLM API Key"]
    default_index = 0 if not openai_api_key else 2
    llm_choice = st.sidebar.radio(
        label="Select LLM for analysis:", 
        options=available_options, 
        index=default_index)
    
    if llm_choice == "GPT-4o":
        return llm_choice, None, None
    elif llm_choice == "Groq":
        return llm_choice, None, None
    elif llm_choice == "OpenAI":
        model, openai_api_key = get_openai_key(openai_api_key)
        return llm_choice, model, openai_api_key
    elif llm_choice == "BambooLLM":
        return llm_choice, None, None
    else:
        bamboollm_key = st.sidebar.text_input(label="Your BambooLLM API Key:", type="password")
        if not bamboollm_key:
            st.error("BambooLLM API key not set for your chat")
            st.stop()
        return "BambooLLM", None, bamboollm_key

def display_example_questions():
    
    with st.sidebar:
        st.divider()
        st.markdown("## Example Questions")
        st.write("tell me something interesting about the dataset in a plot?")
        st.write("summarize the dataset")
        st.write("Are there any missing values in the dataset? If so, which columns have them?")
        st.write("create a histogram for [a specific column]?")
        st.write("provide a scatterplot for [two specific columns]?")

    with st.sidebar:
        st.divider()
        st.markdown("## Important Notes")
        st.markdown("#### The tool can make mistakes!")
        st.markdown("#### The Groq model is llama3-groq-70b-8192-tool-use-preview")
        st.markdown("#### If the answers are not good from Groq and BambooLLM, you could consider OpenAI.")


        

