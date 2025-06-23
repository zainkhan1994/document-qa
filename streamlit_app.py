import streamlit as st
from openai import OpenAI
import pandas as pd
import PyPDF2
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(
    page_title="DocQuery AI",
    page_icon="🤖",
    layout="wide"
)

# --- Custom CSS for a prettier UI ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1y4p8pa {
        padding-top: 2rem;
    }
    .st-emotion-cache-1v0mbdj {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .st-emotion-cache-16txtl3 {
        padding: 1rem;
    }
    .st-emotion-cache-10trblm {
        text-align: center;
        font-family: 'Georgia', serif;
    }
    h1 {
        color: #3A3A3A;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Function to Parse Files ---
def get_document_text(uploaded_file):
    """Extracts text from the uploaded file based on its type."""
    text = ""
    file_type = uploaded_file.type
    
    try:
        if file_type == "text/plain" or file_type == "text/markdown":
            text = uploaded_file.read().decode()
        elif file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
            text = df.to_string()
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
        
    return text

# --- Sidebar for API Key ---
with st.sidebar:
    st.header("🔑 API Configuration")
    st.write("To use this app, you need to provide an OpenAI API key.")
    openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", help="You can get your key from [here](https://platform.openai.com/account/api-keys).")
    
    st.markdown("---")
    st.info("Your API key is handled securely and is not stored.")

# --- Main Application ---
st.title("📄 DocQuery AI: Ask Questions About Your Documents")
st.write(
    "Upload a document, ask a question, and let GPT provide you with the answer. "
    "Supports .txt, .md, .pdf, and .csv files."
)

if not openai_api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.", icon="🗝️")
else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.subheader("1. Upload Your Document")
            uploaded_file = st.file_uploader(
                "Choose a document",
                type=["txt", "md", "pdf", "csv"],
                label_visibility="collapsed"
            )

    with col2:
        with st.container(border=True):
            st.subheader("2. Ask a Question")
            question = st.text_area(
                "Now, ask a question about the uploaded document.",
                placeholder="e.g., Can you give me a short summary of this document?",
                disabled=not uploaded_file,
                label_visibility="collapsed"
            )

    if uploaded_file and question:
        with st.spinner("Analyzing document and generating answer..."):
            # Process the uploaded file and question.
            document_text = get_document_text(uploaded_file)
            
            if document_text:
                messages = [
                    {
                        "role": "user",
                        "content": f"Based on the following document, please answer the question provided.\n\n"
                                   f"**Document:**\n{document_text}\n\n"
                                   f"---\n\n"
                                   f"**Question:** {question}",
                    }
                ]

                try:
                    # Generate an answer using the OpenAI API.
                    stream = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        stream=True,
                    )
                    
                    st.subheader("🤖 AI Generated Answer")
                    # Stream the response to the app.
                    with st.container(border=True):
                        st.write_stream(stream)

                except Exception as e:
                    st.error(f"An error occurred with the OpenAI API: {e}")
