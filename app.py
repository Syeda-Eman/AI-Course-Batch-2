#importing necessary libraries
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import output_parser
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
from gtts.lang import tts_langs
import os

# define the list of available languages for TTS
langs = tts_langs().keys()

# defining API key
api_key = "AIzaSyA3rW22ToS1WW9gDi2wNysw7HjH18EkIZY"

# create a chat template with prompt
chat_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are an AI speech-to-speech question-answering assistant. 
               Your role is to generate responses based on user inputs or queries in the Urdu language.
               Ensure that your answers are clear, concise, and easy to understand. 
               Always maintain a friendly and helpful tone, and respond to user queries exclusively in pure Urdu.'''
        ),
        ("human", "{human_input}"),
    ]
)

# initialize the model with the Google Generative AI
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# Create the chain by combining the template, model, and output parser
chain = chat_template | model | StrOutputParser()

# Streamlit interface
st.title("اردو سوال و جواب کرنے والا بوٹ")
st.write("اردو میں سوال کریں اور اردو میں جواب حاصل کریں۔")

# Convert speech to text
text = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT")

# If text is recognized, generate a response and convert it to speech
if text:
    st.write(f"آپ کا سوال: {text}")  # Display the recognized text in Urdu
    with st.spinner("جواب تیار کیا جا رہا ہے..."):
        res = chain.invoke({"human_input": text})
        
        # Ensure the response is in Urdu
        if not res:
            st.error("جواب فراہم نہیں کیا جا سکا۔ براہ کرم دوبارہ سے کوشش کریں۔")
        else:
            # Convert response text to speech using gTTS
            tts = gTTS(text=res, lang='ur')
            tts.save("response.mp3")
            audio_file = open("response.mp3", "rb")
            audio_bytes = audio_file.read()

            # Display the response text
            st.write(f"جواب: {res}")

            # Display audio player
            st.audio(audio_bytes, format="audio/mp3")
else:
    st.error("معاف کیجیے، میں آپ کے پیغام کو سمجھ نہیں سکا۔ براہ کرم اپنا سوال دوبارہ پوچھیں۔")