import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from app import run_graph_with_user_input  # Your LangGraph flow

st.set_page_config(page_title="LangGraph Multi-Agent Chatbot", layout="wide")
st.title("🤖 Smart Research and Code Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message(name="user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message(name="assistant"):
            st.markdown(msg.content)
    else:
        with st.chat_message(name="assistant"):
            st.markdown(msg.content)

# Input box
user_input = st.chat_input("Ask your question...")

if user_input:
    # Show user message instantly
    with st.chat_message(name="user"):
        st.markdown(user_input)

    # Run LangGraph app
    history = st.session_state.messages
    new_history = run_graph_with_user_input(user_input, history)

    # Extract new messages only
    new_messages = new_history[len(history):]

    for msg in new_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(name=role):
            st.markdown(msg.content)

    # Save to session
    st.session_state.messages = new_history

