import streamlit as st
from transformers import pipeline
from gtts import gTTS
import base64
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Load the translation pipeline
@st.cache_resource
def load_translation_pipeline():
    try:
        translator = pipeline("translation", model="TresorB/TshilubaEnglishTranslationModel")
        return translator
    except Exception as e:
        logger.error(f"Failed to load the translation pipeline: {e}")
        st.error("Failed to load the translation pipeline. Please check the logs for more details.")
        return None

# Streamlit App
st.title("MarianMT Model Translation")

# Load Translation Pipeline
translator = load_translation_pipeline()
if translator:
    st.success("Translation pipeline loaded successfully from Hugging Face.")
else:
    st.error("Failed to load the translation pipeline.")

# Translation interface
st.subheader("Translate Tshiluba to English")

tshiluba_text = st.text_area("Enter Tshiluba text to translate")
if st.button("Translate"):
    if tshiluba_text and translator:
        with st.spinner("Translating..."):
            # Generate translation
            translated_text = translator(tshiluba_text)[0]['translation_text']
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

