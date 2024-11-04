import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import os

# Define your API key (replace with actual API key)
api_key = "AIzaSyCxQtYjg28vBKICtZweHIunusmIPRi5Tts"

# Create a chat template with prompt
chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Please always respond to user queries in Pure Urdu language and use the context from the PDF if available.",
        ),
        ("human", "{human_input}"),
    ]
)

# Initialize the model with the Google Generative AI
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Create the chain by combining the template, model, and output parser
chain = chat_template | model | StrOutputParser()

# Streamlit interface
st.set_page_config(page_title="اردو سوال و جواب بوٹ", page_icon="🤖")
st.title("اردو سوال و جواب بوٹ")
st.write("آپ اردو میں سوال کریں اور جوابات حاصل کریں۔")

# PDF file upload
uploaded_file = st.file_uploader("PDF فائل اپ لوڈ کریں", type=["pdf"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_pdf.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF file
    pdf_loader = PyPDFLoader(temp_file_path)
    documents = pdf_loader.load()

    # Speech input section
    st.subheader("اپنا سوال بولیں:")
    speech_input = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT")

    # Text input for user query
    user_input = st.text_input("یا اپنا سوال یہاں لکھیں (اردو میں):", placeholder="اپنا سوال یہاں لکھیں...")

    # Combine inputs if both are provided
    final_input = speech_input if speech_input else user_input

    if final_input:
        st.write(f"آپ کا سوال: **{final_input}**")  # Display the recognized or typed text in Urdu
        with st.spinner("جواب تیار کیا جا رہا ہے..."):
            # Combine the context from the PDF and the user's question
            context = "\n".join([doc.page_content for doc in documents])  # Extract text from documents
            full_input = f"{context}\n\nUser Question: {final_input}"

            # Invoke the model with the combined input
            res = chain.invoke({"human_input": full_input})

            # Ensure the response is in Urdu
            if not res:
                st.error("جواب پیدا نہیں کیا جا سکا۔ براہ کرم دوبارہ کوشش کریں۔")
            else:
                # Display the response text
                st.subheader("جواب:")
                st.write(f"**{res}**")

                # Convert response text to speech using gTTS
                tts = gTTS(text=res, lang='ur')
                tts_file = "response.mp3"
                tts.save(tts_file)

                # Play the audio response
                with open(tts_file, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")

                # Clean up the audio file after playing
                os.remove(tts_file)

    # Clean up the temporary PDF file
    os.remove(temp_file_path)
else:
    st.info("براہ کرم PDF فائل اپ لوڈ کریں۔")

