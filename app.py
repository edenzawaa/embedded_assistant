from flask import Flask, request, jsonify, Response
import os
import speech_recognition as sr
import requests
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as genai

app = Flask(__name__)
WAV_FILE = 'recording.wav'
RESPONSE_MP3 = 'response.mp3'
RESPONSE_WAV = 'response.wav'

load_dotenv()

# Configure your Gemini / Generative AI API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Pick a Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/testConnection', methods=['GET'])
def test_connection():
    try:
        return jsonify({'status': 'Server is up'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
@app.route('/uploadAudio', methods=['POST'])
def upload_audio():
    try:
        # Save uploaded audio
        with open(WAV_FILE, 'wb') as f:
            f.write(request.data)

        # Transcribe speech to text (Vietnamese)
        transcription = speech_to_text(WAV_FILE, lang='vi-VN')

        # Generate reply using Ollama
        reply = query_gemini(transcription)

        # Convert reply to speech using gTTS (Vietnamese)
        text_to_speech(reply, RESPONSE_MP3)
    
        # Convert MP3 to WAV for compatibility with ESP32 streaming
        AudioSegment.from_mp3(RESPONSE_MP3).export(RESPONSE_WAV, format="wav")

        return jsonify({
            'transcription': transcription,
            'assistant_reply': reply,
            'audio_file': RESPONSE_WAV
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

upload_audio();

def speech_to_text(file_name, lang):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language=lang)
            print(f'Transcription: {text}')
            return text
        except sr.UnknownValueError:
            return "Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"

def query_gemini(prompt: str) -> str:
    try:
        full_prompt = (
            f"Người dùng nói: '{prompt}'. "
            "Vui lòng trả lời bằng tiếng Việt, ngắn gọn dưới 50 từ."
        )
        response = gemini_model.generate_content(full_prompt)
        reply = response.text.strip()
        print("Gemini reply:", reply)
        return reply
    except Exception as e:
        print("Gemini error:", e)
        return "Lỗi khi truy vấn Gemini"
    
def text_to_speech(text, filename):
    # Use gTTS for natural Vietnamese voice
    tts = gTTS(text=text, lang='vi')
    tts.save(filename)

@app.route('/getReplyAudio')
def get_reply_audio():
    def generate_and_cleanup():
        try:
            with open(RESPONSE_WAV, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        finally:
            for f in [RESPONSE_WAV, RESPONSE_MP3, WAV_FILE]:
                if os.path.exists(f):
                    os.remove(f)

    return Response(generate_and_cleanup(), mimetype="audio/wav")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's provided port
    print(f'Listening at {port}')
    app.run(host='0.0.0.0', port=port, threaded=True)

