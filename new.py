import streamlit as st
import openai
import google.generativeai as genai
import requests

import tempfile
import base64

# ----------------------------
# API KEYS (Replace with your real keys)
# ----------------------------
openai.api_key = "your_openai_api_key"
genai.configure(api_key="your_gemini_api_key")
gemini = genai.GenerativeModel("gemini-1.5-pro")  # or "gemini-pro" if needed
chat_model = "gpt-4"  # or "gpt-3.5-turbo"

# ----------------------------
# Streamlit UI Setup
# ----------------------------
st.set_page_config("📘 AI Physics Tutor", layout="centered")
st.title("👩‍🏫 AI-Powered Physics Tutor")
st.markdown("Upload diagrams, use voice, or type your physics question. Choose language and grade level.")

# ----------------------------
# User Inputs
# ----------------------------
language = st.selectbox("🌐 Language", ["English", "Telugu", "Hindi", "Marathi", "Spanish"])
grade = st.selectbox("🎓 Grade Level", [str(i) if i <= 12 else f"College Year {i-12}" for i in range(6, 15)])

question = st.text_area("✍️ Type your question here:")

use_voice = st.checkbox("🎤 Use voice instead")
audio_file = None
if use_voice:
    audio_file = st.file_uploader("Upload your question as audio (.wav/.mp3)", type=["wav", "mp3"])
    if audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        recognizer = sr.Recognizer()
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        try:
            question = recognizer.recognize_google(audio)
            st.success(f"Recognized Question: {question}")
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")

image = st.file_uploader("🖼️ Upload a diagram or handwritten image (optional)", type=["jpg", "jpeg", "png"])

use_chatgpt = st.checkbox("🤖 Use ChatGPT")
use_llama = st.checkbox("🐏 Use LLaMA Finetuned Model")

# ----------------------------
# Prompt Construction
# ----------------------------
prompt_base = f"""
Act as a physics tutor for {grade} students.
Explain in {language} using simple steps and examples.
Now explain: {question}
"""

# ----------------------------
# Answer Generation
# ----------------------------
if st.button("🧠 Get Explanation"):
    if not question:
        st.warning("Please provide a question (typed or via voice).")
    else:
        with st.spinner("AI is thinking..."):
            result_text = ""

            if image:
                bytes_data = image.read()
                gemini_input = [{"mime_type": "image/jpeg", "data": bytes_data}, prompt_base]
                gemini_response = gemini.generate_content(gemini_input)
                result_text = gemini_response.text
            else:
                gemini_response = gemini.generate_content(prompt_base)
                result_text = gemini_response.text

            st.markdown("### 🌟 Gemini Tutor Response")
            st.write(result_text)

            if use_chatgpt:
                st.markdown("### 🤖 ChatGPT Answer")
                try:
                    chat_response = openai.ChatCompletion.create(
                        model=chat_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful Physics tutor."},
                            {"role": "user", "content": prompt_base}
                        ]
                    )
                    st.write(chat_response.choices[0].message.content)
                except Exception as e:
                    st.error(f"OpenAI Error: {e}")

            if use_llama:
                st.markdown("### 🐏 LLaMA Finetuned Answer")
                try:
                    llama_response = requests.post(
                        "https://api-inference.huggingface.co/models/YOUR_FINETUNED_LLAMA_MODEL",
                        headers={"Authorization": "Bearer YOUR_HUGGINGFACE_TOKEN"},
                        json={"inputs": prompt_base}
                    )
                    st.write(llama_response.json()[0]["generated_text"])
                except Exception as e:
                    st.error(f"LLaMA Error: {e}")
