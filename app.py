import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import base64
import os
import logging
import requests
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Hugging Face model and tokenizer URLs
model_url = 'https://huggingface.co/TresorB/TshilubaEnglishTranslationModel/resolve/main/New_best_model.zip'
tokenizer_url = 'https://huggingface.co/TresorB/TshilubaEnglishTranslationTokenizer/resolve/main/New_best_tokenizer.zip'

# Hugging Face token for authentication
hf_token = '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'

# Function to download a file from Hugging Face
def download_file(url, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Check for HTTP errors
    return BytesIO(response.content)

# Function to extract and load model and tokenizer
def load_from_zip(zip_bytes, extract_to):
    import zipfile
    import os
    
    with zipfile.ZipFile(zip_bytes) as z:
        z.extractall(extract_to)

# Function to load the model from Hugging Face
@st.cache_resource
def load_model():
    model_zip = download_file(model_url, hf_token)
    model_path = "./model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        load_from_zip(model_zip, model_path)
    try:
        model = MarianMTModel.from_pretrained(model_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error("Failed to load the model. Please check the logs for more details.")
        return None

# Function to load the tokenizer from Hugging Face
@st.cache_resource
def load_tokenizer():
    tokenizer_zip = download_file(tokenizer_url, hf_token)
    tokenizer_path = "./tokenizer"
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)
        load_from_zip(tokenizer_zip, tokenizer_path)
    try:
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        st.error("Failed to load the tokenizer. Please check the logs for more details.")
        return None

# Streamlit App
st.title("MarianMT Model Translation")

# Load Model and Tokenizer
model = load_model()
tokenizer = load_tokenizer()
if model and tokenizer:
    st.success("Model and Tokenizer loaded successfully from Hugging Face.")
else:
    st.error("Failed to load Model and Tokenizer.")

# Translation interface
st.subheader("Translate Tshiluba to English")

tshiluba_text = st.text_area("Enter Tshiluba text to translate")
if st.button("Translate"):
    if tshiluba_text and model and tokenizer:
        with st.spinner("Translating..."):
            # Tokenize input
            inputs = tokenizer(tshiluba_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

            # Generate translation
            translated = model.generate(**inputs)

            # Decode the output
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success(f"Translated text: {translated_text}")
            
            # Convert translated text to speech
            tts = gTTS(translated_text)
            tts.save("translated_audio.mp3")

            # Display audio player
            audio_file = open("translated_audio.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

            # Optionally provide a download link
            def get_binary_file_downloader_html(bin_file, file_label='File'):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                bin_str = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
                return href

            st.markdown(get_binary_file_downloader_html("translated_audio.mp3", 'Download translated audio'), unsafe_allow_html=True)
    else:
        st.warning("Please enter some Tshiluba text to translate.")
