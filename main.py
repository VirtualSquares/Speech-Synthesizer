from flask import Flask, render_template, jsonify
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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fact_check', methods=['POST'])
def fact_check():
    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 120)

    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        cText = recognizer.recognize_google(audio)
        print("You said:", cText)  # Debug output

        if not cText:
            return jsonify({'error': "No text provided for summarization."})

        response = model.generate_content("Summarize this into a shorter version through multiple bullet points, max of 10, this is the bullet point you should use: • " + cText)
        response_text = response.text

        print("Response from model:", response_text)  # Debug output

        if not response_text:
            return jsonify({'error': "The model did not return any response."})

        tts_engine.say(response_text)
        tts_engine.runAndWait()

        return jsonify({'response': response_text})

    except sr.UnknownValueError:
        return jsonify({'error': "Sorry, I could not understand the audio."})
    except sr.RequestError as e:
        return jsonify({'error': f"Could not request results from Google Speech Recognition service; {e}"})
    except Exception as e:
        return jsonify({'error': f"An error occurred: {e}"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

