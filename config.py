import streamlit as st
import openai

supported_chat_models = [
    "gpt-4o-mini", 
    "gpt-4",  
    "gpt-4o", 
    "gpt-3.5-turbo",
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
    available_options = ["Groq", "GPT-4o-mini", "GPT-4o", "BambooLLM", "OpenAI", "Your BambooLLM API Key"]
    default_index = 0 if not openai_api_key else 2
    llm_choice = st.sidebar.radio(
        label="Select LLM for analysis:", 
        options=available_options, 
        index=default_index)
    
    if llm_choice == "GPT-4o":
        return llm_choice, None, None
    elif llm_choice == "GPT-4o-mini":
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
        st.write("You can get your free API key for BambooLLM or Groq signing up at https://pandas-ai.com or https://groq.com")
        st.divider()
        st.markdown("## Click on an Example Question to Try the App")
    #     example_questions = [
    #         "Tell me something interesting about the dataset in a plot?",
    #         "Summarize the dataset",
    #         "Are there any missing values in the dataset? If so, which columns have them?",
    #         "Create a histogram for any column?",
    #         "Provide a scatterplot for any two columns?"
    #     ]

    #     if "clicked_question" not in st.session_state:
    #         st.session_state.clicked_question = None
        
    #     for question in example_questions:
    #         col = st.columns(1)
    #         with col[0]:
    #             if st.button(question):
    #                 st.session_state.clicked_question = question
    # return st.session_state.clicked_question
        st.markdown("""
            <style>
            .stButton > button {
                background-color: #dcf1f7;
                color: black; 
                border: none;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 5px 0;
                cursor: pointer;
                width: 100%;
                border-radius: 10px;
                transition-duration: 0.4s;
            }
            .stButton > button:hover {
                background-color: #D3D3D3;
                color: black;
            }
            </style>
        """, unsafe_allow_html=True)

        example_questions = [
            "Tell me something interesting about the dataset in a plot?",
            "Summarize the dataset",
            "Are there any missing values in the dataset? If so, which columns have them?",
            "Create a histogram for any column?",
            "Provide a scatterplot for any two columns?"
        ]

        selected_question = None
        # Create buttons using a for loop to ensure consistent style and layout
        for idx, question in enumerate(example_questions):
            if st.button(question, key=f"btn_{idx}"):
                selected_question = question

    return selected_question

def display_notes():
    with st.sidebar:
        st.divider()
        st.write("You can get your free API key for BambooLLM igning up at https://pandas-ai.com ")
        st.divider()
        st.markdown("## Important Notes")
        st.markdown("#### The tool can make mistakes!")
        st.markdown("#### The Groq model is llama-3.2-90b-text-preview")
        st.markdown("#### If the answers are not good from Groq and BambooLLM, you could consider OpenAI.")


        

