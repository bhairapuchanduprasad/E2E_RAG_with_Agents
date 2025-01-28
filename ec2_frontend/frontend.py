import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()

# Backend API Endpoint
API_URL = os.getenv("API_URL")

st.title("HR_Chat: Your Personal HR Assistant MVP")
st.markdown("Ask your HR-related questions, and I'll provide you with the most relevant answers based on the internal data!")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_chat_history" not in st.session_state:
    st.session_state.show_chat_history = False
if "retrieved_contexts" not in st.session_state:
    st.session_state.retrieved_contexts = []
if "show_retrieved_contexts" not in st.session_state:
    st.session_state.show_retrieved_contexts = False
if "evaluation" not in st.session_state:
    st.session_state.evaluation = []
if "show_evaluation" not in st.session_state:
    st.session_state.show_evaluation = False


# Sidebar: Toggle Chat History
if st.sidebar.button("Chat History"):
    st.session_state.show_chat_history = not st.session_state.show_chat_history
if st.session_state.show_chat_history:
    st.sidebar.subheader("Chat History")
    st.sidebar.json(st.session_state.messages)


# Sidebar: Toggle Retrieved Contexts
if st.sidebar.button("Retrieved Contexts"):
    st.session_state.show_retrieved_contexts = not st.session_state.show_retrieved_contexts
if st.session_state.show_retrieved_contexts:
    st.sidebar.subheader("Retrieved Contexts")
    st.sidebar.json(st.session_state.retrieved_contexts)


# Sidebar: Toggle Retrieved Contexts
if st.sidebar.button("Evaluation"):
    st.session_state.show_evaluation = not st.session_state.show_evaluation
if st.session_state.show_evaluation:
        st.sidebar.subheader("RAGAS Evaluation")
        st.sidebar.write("Evaluating.....")
        appended_content = []
        try:
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    appended_content.append(message["content"])
            if appended_content:
                latest_user_query = appended_content[-1]
            response = requests.post(API_URL, json={"question": latest_user_query, "evaluation": "yes", "top_k": 3})
            response_data = response.json()
            st.sidebar.write(response_data)
            st.sidebar.json({"query": latest_user_query, "answer": response_data.get("answer", "No answer found"), "evaluation_results" : response_data.get("evaluation") })
        except Exception as e:
            st.sidebar.write( f"Error Evaluating: {str(e)}")

# Display Chat History
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["classification"])
            st.markdown(message["content"])



# Chat Input
if user_query := st.chat_input("Please ask your HR queries here..."):
    # Add User Message to Chat History
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Backend Interaction
    with st.spinner("Processing your query..."):
        try:
            # POST request to backend
            response = requests.post(API_URL, json={"question": user_query, "top_k": 3})
            # Check for successful response
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data.get("answer", "No answer found.")
            else:
                answer = "Sorry, an error occurred while processing your query."

        except Exception as e:
            answer = f"Error connecting to the backend: {str(e)}"

    # append Assistant Response to Chat History
    st.session_state.messages.append({"role": "assistant", "classification": response_data.get("classification", "No answer found"), "content": answer})
    st.session_state.retrieved_contexts.append({"query": user_query, "classification": response_data.get("classification", "No answer found"), "content": response_data.get("chunks", "No answer found")})

    st.rerun()
