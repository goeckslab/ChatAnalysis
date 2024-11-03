import streamlit as st
import openai

supported_chat_models = [
    "gpt-4o-mini", 
    "gpt-4",  
    "gpt-4o", 
    "gpt-3.5-turbo",
]

groq_models = [
    "llama-3.2-90b-text-preview",
    "llama-3.2-90b-vision-preview",
    "llama-3.1-70b-versatile",
]

def get_groq_key(groq_api_key_user):
    if not groq_api_key_user:
        groq_api_key_user = st.sidebar.text_input(label="Your Groq API Key:", type="password")
    if not groq_api_key_user:
        st.error("Groq API key not set for your chat")
        st.stop()
    
    model = st.sidebar.selectbox(label="Select the model you want", options=groq_models)
    return model, groq_api_key_user

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

def configure_llm_options(openai_api_key, groq_api_key_user):
    available_options = ["llama-3.2-90b", "GPT-4o-mini", "GPT-4o", "BambooLLM", "OpenAI", "Your-BambooLLM-API-Key", "Your-Groq-API-Key"]
    default_index = 0 
    if groq_api_key_user:
        default_index = 6
    elif openai_api_key:
        default_index = 4

    llm_choice = st.sidebar.radio(
        label="Select a LLM for analysis (You are encouraged to use your own API keys. See below for more information):", 
        options=available_options, 
        index=default_index)
    
    if llm_choice == "GPT-4o":
        return llm_choice, None, None
    elif llm_choice == "GPT-4o-mini":
        return llm_choice, None, None
    elif llm_choice == "llama-3.2-90b":
        return "Groq", None, None
    elif llm_choice == "OpenAI":
        model, openai_api_key = get_openai_key(openai_api_key)
        return llm_choice, model, openai_api_key
    elif llm_choice == "BambooLLM":
        return llm_choice, None, None
    elif llm_choice == "Your-BambooLLM-API-Key":
        bamboollm_key = st.sidebar.text_input(label="Your BambooLLM API Key:", type="password")
        if not bamboollm_key:
            st.error("BambooLLM API key not set for your chat")
            st.stop()
        return "BambooLLM", None, bamboollm_key
    elif llm_choice == "Your-Groq-API-Key":
        model, groq_api_key_user = get_groq_key(groq_api_key_user)
        return llm_choice, model, groq_api_key_user

def display_example_questions():
    with st.sidebar:
        st.divider()
        st.write("You can get your free API key for BambooLLM by signing up at https://pandas-ai.com and your free API key for Groq by signing up at https://groq.com.")
        
        
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
            "Delete the columns where most of the values are missing and return the modified dataset",
            "Return the types of columns in a dataframe.",
            "Provide the statistics for each column in a properly formatted DataFrame. The DataFrame must include a column named 'column_name' that contains the name of each column.",
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
        st.markdown("## Important Notes")
        st.markdown("#### The tool can make mistakes!")
        st.markdown("#### If the answers are not satisfactory from one LLM, you could consider switching to another LLM or asking a more specific question.")


        

