import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from app import run_graph_with_user_input  # Import from your LangGraph code

st.set_page_config(page_title="LangGraph Multi-Agent Chatbot", layout="wide")
st.title("Smart Research and Code Bot")

# Initialize session state memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("ai").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

# Chat input box
user_input = st.chat_input("Ask your question...")

if user_input:
    # Display user input immediately
    st.chat_message("user").write(user_input)

    # Run through LangGraph
    history = st.session_state.messages
    new_history = run_graph_with_user_input(user_input, history)

    # Extract the new AI message
    new_messages = new_history[len(history):]
    for msg in new_messages:
        st.chat_message(msg.name if hasattr(msg, "name") else "ai").write(msg.content)

    # Update session memory
    st.session_state.messages = new_history
