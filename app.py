import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import base64
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face token
hf_token = "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"

# Hugging Face model and tokenizer names
model_name = 'TresorB/TshilubaEnglishTranslationModel'
tokenizer_name = 'TresorB/TshilubaEnglishTranslationTokenizer'

# Load model and tokenizer with token
@st.cache_resource
def load_model():
    try:
        model = MarianMTModel.from_pretrained(model_name, use_auth_token=hf_token)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error("Failed to load the model. Please check the logs for more details.")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_name, use_auth_token=hf_token)
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
            inputs = tokenizer(tshiluba_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success(f"Translated text: {translated_text}")

            # Convert translated text to speech
            tts = gTTS(translated_text)
            tts.save("translated_audio.mp3")

            # Display audio player
            audio_file = open("translated_audio.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

            # Provide download link for audio
            def get_binary_file_downloader_html(bin_file, file_label='File'):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                bin_str = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
                return href

            st.markdown(get_binary_file_downloader_html("translated_audio.mp3", 'Download translated audio'), unsafe_allow_html=True)
    else:
        st.warning("Please enter some Tshiluba text to translate.")

