import streamlit as st
import base64
import os

# Page configuration
st.set_page_config(
    page_title="Echofix Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to add background image
def add_bg_image():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to the image
    image_path = os.path.join(script_dir, "bg.png")
    
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    
    return f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{b64_encoded});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        margin: 0;
        padding: 0;
    }}
    </style>
    """

# Main function
def main():
    # Hide default elements
    hide_elements = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Disable scrolling */
    body {
        overflow: hidden;
    }
    
    /* Center button */
    .center-div {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    
    .chat-button {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    
    .chat-button:hover {
        background-color: #3e8e41;
        box-shadow: 0 12px 20px 0 rgba(0,0,0,0.3);
    }
    </style>
    
    <div class="center-div">
        <a href="/1_Chatbot" class="chat-button">Chat with Support Assistant</a>
    </div>
    """
    
    # Add background image
    st.markdown(add_bg_image(), unsafe_allow_html=True)
    
    # Add centered button
    st.markdown(hide_elements, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
