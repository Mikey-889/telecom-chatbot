import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Set Hugging Face credentials (for Streamlit Cloud)
HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]  # Store in Streamlit secrets
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# FAQ Data
FAQ_DATA = {
    "SIM Services": {
        "sim_activation": {
            "query": ["sim not activated", "sim not working", "new sim not working"],
            "response": "Here are the steps to activate your SIM:\n1. Insert SIM in your phone\n2. Wait for 30 minutes\n3. Restart your phone\n4. Dial *282# to check activation status\nIf still not working, ensure you've completed the KYC process.",
            "quick_reply": "SIM not activated"
        },
        "sim_replacement": {
            "query": ["replace sim", "damaged sim", "lost sim"],
            "response": "To replace your SIM card:\n1. Visit nearest store with ID proof\n2. Fill out SIM replacement form\n3. Pay replacement fee\n4. Get new SIM with same number",
            "quick_reply": "Replace SIM"
        }
    },
    "Internet Services": {
        "broadband_issues": {
            "query": ["broadband not working", "internet down", "no internet connection"],
            "response": "Please follow these troubleshooting steps:\n1. Check all cable connections\n2. Restart your router\n3. Verify if the issue is in your area\n4. Check router lights status\nContact support if issue persists.",
            "quick_reply": "Broadband issues"
        }
    }
}

def load_model():
    """Load the AI model with authentication."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=HUGGINGFACE_TOKEN)
    return model, tokenizer

def get_response(query, category=None):
    """Fetch response from FAQ or return 'Out of context'."""
    query = query.lower()

    if category and category != "All Categories":
        category_data = FAQ_DATA.get(category, {})
    else:
        category_data = {key: value for category in FAQ_DATA.values() for key, value in category.items()}

    for key, data in category_data.items():
        if any(q in query for q in data["query"]):
            return data["response"]

    return "Out of context"

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "All Categories"

def main():
    st.set_page_config(page_title="Telecom Support Assistant", layout="wide")

    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.header("Filters")
        categories = ["All Categories"] + list(FAQ_DATA.keys())
        selected_category = st.selectbox("Select Category", categories)
        st.session_state.selected_category = selected_category

        # Quick replies
        st.header("Quick Replies")
        category_data = FAQ_DATA.get(selected_category, FAQ_DATA) if selected_category != "All Categories" else FAQ_DATA
        for category in category_data.values():
            for item in category.values():
                if st.button(item["quick_reply"], key=item["quick_reply"]):
                    new_message = item["query"][0]
                    st.session_state.messages.append({"role": "user", "content": new_message})
                    response = get_response(new_message, selected_category)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # Chat Interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input
    if prompt := st.chat_input("Type your query here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = get_response(prompt, st.session_state.selected_category)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear Chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
