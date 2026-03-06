import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"  # FastAPI endpoint

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.caption("Ask questions from your knowledge base")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
user_input = st.chat_input("Ask your question...")

if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = requests.post(
                API_URL,
                json={"question": user_input}
            )

            if response.status_code == 200:
                answer = response.json()["answer"]
            else:
                answer = "Error: Could not connect to API"

            st.markdown(answer)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": answer})
