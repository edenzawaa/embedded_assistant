from flask import Flask, request, jsonify, Response
import os
import speech_recognition as sr
import requests
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as genai
import time
import concurrent.futures
import threading
import traceback

app = Flask(__name__)
WAV_FILE = 'recording.wav'
RESPONSE_MP3 = 'response.mp3'
RESPONSE_WAV = 'response.wav'

load_dotenv()

# Configure your Gemini / Generative AI API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Pick a Gemini model
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

@app.route('/uploadAudio', methods=['POST'])
def upload_audio():
    try:
        print(">> [UPLOAD] Received audio upload.")

        # Save uploaded audio
        with open(WAV_FILE, 'wb') as f:
            f.write(request.data)
        print(">> [UPLOAD] Saved WAV file.")

        # Start background processing so we can reply immediately
        threading.Thread(target=process_audio_and_generate_reply).start()

        # Immediate response to ESP32 — avoids long blocking
        return jsonify({'status': 'processing'}), 202

    except Exception as e:
        print(">> [ERROR] upload_audio:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def process_audio_and_generate_reply():
    """Runs in background: transcribe, query Gemini, TTS, and export WAV."""
    try:
        print(">> [THREAD] Starting background processing...")

        # Step 1. Transcribe
        transcription = speech_to_text(WAV_FILE, lang='vi-VN')
        print(f">> [THREAD] Transcription: {transcription}")

        # Step 2. Query Gemini
        reply = query_gemini(transcription)
        print(f">> [THREAD] Gemini reply: {reply}")

        # Step 3. Convert reply to MP3
        text_to_speech(reply, RESPONSE_MP3)
        print(">> [THREAD] gTTS done, converting to WAV...")

        # Step 4. Convert MP3 → WAV
        AudioSegment.from_mp3(RESPONSE_MP3).export(RESPONSE_WAV, format="wav")
        print(">> [THREAD] Conversion to WAV complete.")

        print(">> [THREAD] Processing complete — ready for client fetch!")

    except Exception as e:
        print(">> [ERROR] process_audio_and_generate_reply:", e)
        traceback.print_exc()



def speech_to_text(file_name, lang):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    recognizer.recognize_google, audio_data, language=lang
                )
                return future.result(timeout=10)  # 10-second timeout
        except concurrent.futures.TimeoutError:
            return "Speech recognition timed out"
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
    # Wait until response.wav is ready (up to 10 seconds)
    wait_time = 0
    max_wait = 10
    while not os.path.exists(RESPONSE_WAV) and wait_time < max_wait:
        print(f"Waiting for {RESPONSE_WAV}... {wait_time}s")
        time.sleep(0.5)
        wait_time += 0.5

    if not os.path.exists(RESPONSE_WAV):
        print("Error: response.wav not found after waiting.")
        return jsonify({"error": "Audio file not ready"}), 404

    def generate_and_cleanup():
        try:
            with open(RESPONSE_WAV, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
        finally:
            # Cleanup temporary files after streaming
            for fpath in [RESPONSE_WAV, RESPONSE_MP3, WAV_FILE]:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                        print(f"Removed: {fpath}")
                    except Exception as e:
                        print(f"Error removing {fpath}: {e}")

    print("Streaming response.wav to client...")
    return Response(generate_and_cleanup(), mimetype="audio/wav")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use Render's provided port
    print(f'Listening at {port}')
    app.run(host='0.0.0.0', port=port, threaded=True)

