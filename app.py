import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import base64
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize Hugging Face authentication
@st.cache_resource
def init_auth():
    login(token=st.secrets["HUGGING_FACE_TOKEN"])

# Load model with device_map and quantization for faster inference
@st.cache_resource
def load_model(model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True  # Use 4-bit quantization for faster inference
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto"
        )
        return pipe
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Load Sentence Transformer for RAG
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Telecom Knowledge Base for RAG
telecom_knowledge_base = [
    {"question": "How do I activate my SIM?", "answer": "To activate your SIM, insert it into your phone, wait for 30 minutes, and restart your device. Dial *282# to check activation status."},
    {"question": "What should I do if my broadband is not working?", "answer": "1. Check all cable connections. 2. Restart your router. 3. Verify if the issue is in your area. 4. Contact support if the problem persists."},
    {"question": "How can I check my data balance?", "answer": "Dial *123# to check your data balance or use the mobile app."},
    {"question": "How do I recharge my prepaid plan?", "answer": "You can recharge using the mobile app, website, or by visiting a retail store."},
    {"question": "What should I do if I lose my SIM card?", "answer": "Immediately contact customer support to block your SIM and request a replacement."},
    {"question": "How do I troubleshoot slow internet speed?", "answer": "1. Run a speed test. 2. Check connected devices. 3. Clear router cache. 4. Try changing DNS servers."},
    {"question": "How do I pay my bill online?", "answer": "Log in to the customer portal, go to the 'Billing' section, and follow the payment instructions."},
    {"question": "What are the steps to port my number?", "answer": "1. Send an SMS to 1900 with 'PORT <your number>'. 2. Visit the new operator's store with your ID proof. 3. Complete the KYC process."},
    {"question": "How do I activate international roaming?", "answer": "1. Ensure your plan supports international roaming. 2. Activate roaming via the mobile app or customer portal. 3. Check roaming rates before traveling."},
    {"question": "How do I reset my voicemail password?", "answer": "Dial the voicemail access number, follow the prompts, and reset your password."},
    {"question": "How do I check my call history?", "answer": "You can check your call history via the mobile app or by dialing *121#."},
    {"question": "How do I set up call forwarding?", "answer": "Dial *21*<phone number># to activate call forwarding. Dial #21# to deactivate it."},
    {"question": "How do I change my broadband plan?", "answer": "Log in to the customer portal, go to 'Plans', and select a new plan. Confirm the change."},
    {"question": "How do I report a network issue?", "answer": "Call customer support or use the mobile app to report the issue. Provide details like location and problem description."},
]

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
        },
        "international_roaming": {
            "query": ["international roaming", "roaming activation", "travel abroad"],
            "response": "To activate international roaming:\n1. Ensure your plan supports roaming\n2. Activate roaming via the mobile app\n3. Check roaming rates before traveling",
            "quick_reply": "Activate roaming"
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
        },
        "wifi_setup": {
            "query": ["wifi setup", "configure wifi", "change wifi password"],
            "response": "To set up Wi-Fi:\n1. Connect to the router via Ethernet\n2. Access router settings (usually 192.168.1.1)\n3. Configure Wi-Fi name and password\n4. Save settings",
            "quick_reply": "Wi-Fi setup"
        }
    },
    "Billing": {
        "bill_inquiry": {
            "query": ["bill details", "check bill", "billing issue"],
            "response": "To check your bill:\n1. Login to customer portal\n2. Go to 'Billing' section\n3. View current and past bills\n4. Download bill as PDF",
            "quick_reply": "Check bill"
        },
        "payment_issues": {
            "query": ["payment failed", "bill not paid", "payment error"],
            "response": "If your payment failed:\n1. Check your payment method\n2. Ensure sufficient balance\n3. Retry payment\n4. Contact support if issue persists",
            "quick_reply": "Payment issues"
        }
    },
    "Voice Services": {
        "call_forwarding": {
            "query": ["call forwarding", "forward calls", "divert calls"],
            "response": "To set up call forwarding:\n1. Dial *21*<phone number>#\n2. Press call to activate\n3. Dial #21# to deactivate",
            "quick_reply": "Call forwarding"
        },
        "voicemail": {
            "query": ["voicemail setup", "check voicemail", "reset voicemail"],
            "response": "To set up voicemail:\n1. Dial the voicemail access number\n2. Follow the prompts\n3. Set a password\n4. Record a greeting",
            "quick_reply": "Voicemail setup"
        }
    }
}

# Greetings and General Messages
GREETINGS = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! How can I help you with your telecom services?",
    "bye": "Goodbye! If you have more questions, feel free to ask.",
    "thanks": "You're welcome! Let me know if you need further assistance.",
}

# Function to check if query is telecom-related
def is_query_in_context(query):
    # Check if query contains any telecom-related keywords
    telecom_keywords = [
        "sim", "network", "phone", "call", "data", "internet", "bill", 
        "plan", "recharge", "balance", "broadband", "wifi", "connection",
        "signal", "coverage", "router", "modem"
    ]
    return any(keyword in query.lower() for keyword in telecom_keywords)

# RAG-based response generation
def rag_response(query, knowledge_base, model, top_k=3):
    query_embedding = model.encode(query)
    knowledge_embeddings = [model.encode(item["question"]) for item in knowledge_base]
    similarities = util.cos_sim(query_embedding, knowledge_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    responses = [knowledge_base[i]["answer"] for i in top_indices]
    return "\n\n".join(responses)

def get_response(query, category=None):
    query = query.lower()
    
    # Handle greetings and general messages
    for key, response in GREETINGS.items():
        if key in query:
            return response
    
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
    
    # If no match found in FAQ, use RAG for telecom-specific response
    try:
        sentence_model = load_sentence_transformer()
        rag_answer = rag_response(query, telecom_knowledge_base, sentence_model)
        return rag_answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Set background image with full fit
def set_bg(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode()
    st.markdown(
        f"""<style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            font-size: 16px;
            color: #333;
        }}
        .stTextInput>div>div>input {{
            font-size: 16px;
            color: #333;
        }}
        .stButton>button {{
            font-size: 16px;
            color: #333;
        }}
        </style>""",
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Echofix Support Assistant", layout="wide")
    init_auth()

    # Set background image (replace with your image path)
    set_bg("image.png")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # Sidebar for category selection and quick replies
    with st.sidebar:
        st.header("Filters")
        categories = ["All Categories"] + list(FAQ_DATA.keys())
        selected_category = st.selectbox("Select Category", categories)

        st.header("FAQs")
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
    st.title("🤖 Echofix Support Assistant")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input handling
    if prompt := st.chat_input("Type your query..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.processing = True
        st.rerun()

    # Response generation handling
    if st.session_state.processing:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "user":
            with st.spinner("Analyzing your query..."):
                response = get_response(last_message["content"], selected_category)
                st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.processing = False
            st.rerun()

if __name__ == "__main__":
    main()
