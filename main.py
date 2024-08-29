
import speech_recognition as sr
import google.generativeai as genai
import pyttsx3
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("API key for Google Generative AI is not set in environment variables.")

genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

convo = model.start_chat(history=[])

recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

tts_engine.setProperty('rate', 120)

with sr.Microphone() as source:
    print("Listening...")
    audio = recognizer.listen(source)

try:
    cText = recognizer.recognize_google(audio)
    print("You said:", cText)
    
    response = model.generate_content("Summarize this in a shortened form: " + cText)
    response_text = response.text
    print("Response from model:", response_text)
    
    tts_engine.say(response_text)
    tts_engine.runAndWait()

except sr.UnknownValueError:
    print("Sorry, I could not understand the audio.")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
except Exception as e:
    print(f"An error occurred: {e}")
