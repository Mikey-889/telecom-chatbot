import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from datetime import datetime
from huggingface_hub import login
import os
import base64

# Add Hugging Face token to environment
HUGGING_FACE_TOKEN = st.secrets["HUGGING_FACE_TOKEN"] if "HUGGING_FACE_TOKEN" in st.secrets else "your-token-here"

# Initialize Hugging Face authentication
@st.cache_resource
def init_auth():
    try:
        login(HUGGING_FACE_TOKEN)
        return True
    except Exception as e:
        st.error(f"Failed to authenticate with Hugging Face: {str(e)}")
        return False

# Enhanced FAQ data with categories
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
        },
        "speed_issues": {
            "query": ["slow internet", "buffering", "low speed"],
            "response": "To improve internet speed:\n1. Run speed test at speedtest.net\n2. Check connected devices\n3. Clear router cache\n4. Try different DNS servers",
            "quick_reply": "Slow internet"
        }
    },
    "Billing": {
        "bill_inquiry": {
            "query": ["bill details", "check bill", "billing issue"],
            "response": "To check your bill:\n1. Login to customer portal\n2. Go to 'Billing' section\n3. View current and past bills\n4. Download bill as PDF",
            "quick_reply": "Check bill"
        }
    }
}

@st.cache_resource
def load_model():
    try:
        # Using a different model (you can change this to any model you prefer)
        model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # or any other model you want to use
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None

def is_query_in_context(query):
    # Check if query contains any telecom-related keywords
    telecom_keywords = [
        "sim", "network", "phone", "call", "data", "internet", "bill", 
        "plan", "recharge", "balance", "broadband", "wifi", "connection",
        "signal", "coverage", "router", "modem"
    ]
    return any(keyword in query.lower() for keyword in telecom_keywords)

def get_response(query, category=None):
    query = query.lower()
    
    # First check if query is in context
    if not is_query_in_context(query):
        return "I apologize, but this query appears to be outside the scope of telecom support. Please ask questions related to telecom services, billing, or technical support."
    
    # If category is selected, search only in that category
    if category and category != "All Categories":
        category_data = FAQ_DATA[category]
        for key, data in category_data.items():
            if any(q in query for q in data["query"]):
                return data["response"]
    else:
        # Search all categories
        for category_data in FAQ_DATA.values():
            for key, data in category_data.items():
                if any(q in query for q in data["query"]):
                    return data["response"]
    
    # If no match found in FAQ, use the model for telecom-specific response
    try:
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            return "I apologize, but I'm currently unable to process your request. Please try one of the quick replies or rephrase your question."
        
        # Create a telecom-specific context
        context = "You are a telecom customer service agent. Please provide a helpful response to: " + query
        inputs = tokenizer(context, return_tensors="pt", max_length=100, truncation=True)
        
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add a disclaimer for non-FAQ responses
        return response + "\n\nNote: This is a general response. For specific issues, please contact our customer support."
    
    except Exception as e:
        return "I apologize, but I'm having trouble generating a response. Please try using one of the quick replies or contact our support team."

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = "All Categories"

def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Telecom Support Assistant", layout="wide")
    
    # Initialize authentication
    init_auth()
    
    # Set background image
    set_background_image("your_image.jpg")  # Replace with your image path

    # Custom CSS
    st.markdown("""
        <style>
        .stTextInput>div>div>input {
            border-radius: 20px;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .bot-message {
            background-color: #f5f5f5;
        }
        .quick-reply-button {
            margin: 0.2rem;
            padding: 0.5rem 1rem;
            border-radius: 15px;
            border: 1px solid #ddd;
            background-color: white;
            cursor: pointer;
        }
        .quick-reply-button:hover {
            background-color: #e3f2fd;
        }
        </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # Header
    st.title("🤖 Telecom Support Assistant")
    
    # Sidebar with category filter
    with st.sidebar:
        st.header("Filters")
        categories = ["All Categories"] + list(FAQ_DATA.keys())
        selected_category = st.selectbox("Select Category", categories)
        st.session_state.selected_category = selected_category

        # Quick replies based on selected category
        st.header("Quick Replies")
        if selected_category == "All Categories":
            for category in FAQ_DATA.values():
                for item in category.values():
                    if st.button(item["quick_reply"], key=item["quick_reply"]):
                        new_message = item["query"][0]
                        st.session_state.messages.append({"role": "user", "content": new_message})
                        response = get_response(new_message, selected_category)
                        st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            for item in FAQ_DATA[selected_category].values():
                if st.button(item["quick_reply"], key=item["quick_reply"]):
                    new_message = item["query"][0]
                    st.session_state.messages.append({"role": "user", "content": new_message})
                    response = get_response(new_message, selected_category)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # Chat interface
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

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
