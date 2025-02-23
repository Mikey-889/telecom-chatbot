import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from huggingface_hub import login
import base64

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
        pipe = load_model()
        if pipe is None:
            return "I apologize, but I'm currently unable to process your request. Please try one of the quick replies or rephrase your question."
        
        # Create a telecom-specific context
        prompt = f"""<s>[INST] You are a telecom support assistant. 
        Answer this query concisely: {query} [/INST]"""
        
        response = pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            return_full_text=False
        )[0]['generated_text']
        
        return response
        
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
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
        }}
        </style>""",
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="Telecom Support Assistant", layout="wide")
    init_auth()

    # Set background image (replace with your image path)
    set_bg("sample.jpg")

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
    st.title("🤖 Telecom Support Assistant")
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
