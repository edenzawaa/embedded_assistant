from flask import Flask, request, jsonify, Response
import os
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as genai
import time
import concurrent.futures
import threading

app = Flask(__name__)

WAV_FILE = 'recording.wav'
RESPONSE_MP3 = 'response.mp3'
RESPONSE_WAV = 'response.wav'

load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


@app.route('/uploadAudio', methods=['POST'])
def upload_audio():
    try:
        print(">> Received audio upload.")
        with open(WAV_FILE, 'wb') as f:
            f.write(request.data)
        print(">> Saved WAV file.")

        # Process in background
        threading.Thread(target=process_audio).start()

        return jsonify({
            'status': 'processing',
            'message': 'Audio received, processing in background...'
        }), 200

    except Exception as e:
        print(">> ERROR in upload_audio:", e)
        return jsonify({'error': str(e)}), 500


def process_audio():
    try:
        print(">> Transcribing audio...")
        transcription = speech_to_text(WAV_FILE, lang='vi-VN')
        print(f">> Transcription result: '{transcription}'")

        # Fallback if no speech detected
        if not transcription.strip() or "could not" in transcription.lower() or "error" in transcription.lower():
            print(">> No valid transcription detected, using fallback.")
            reply = "Xin lỗi, tôi không nghe rõ. Bạn có thể nói lại không?"
        else:
            reply = query_gemini(transcription)

        print(f">> Gemini reply: {reply}")

        print(">> Generating TTS...")
        text_to_speech(reply, RESPONSE_MP3)
        AudioSegment.from_mp3(RESPONSE_MP3).export(RESPONSE_WAV, format="wav")
        print(">> Response audio ready!")

    except Exception as e:
        print(">> ERROR in process_audio:", e)
        # Still generate a fallback audio
        fallback = "Xin lỗi, đã xảy ra lỗi khi xử lý âm thanh."
        text_to_speech(fallback, RESPONSE_MP3)
        AudioSegment.from_mp3(RESPONSE_MP3).export(RESPONSE_WAV, format="wav")


def speech_to_text(file_name, lang):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        audio_data = recognizer.record(source)
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    recognizer.recognize_google, audio_data, language=lang
                )
                return future.result(timeout=10)
        except concurrent.futures.TimeoutError:
            return "Speech recognition timed out"
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"


def query_gemini(prompt: str) -> str:
    try:
        full_prompt = (
            f"Người dùng nói: '{prompt}'. "
            "Vui lòng trả lời bằng tiếng Việt, ngắn gọn dưới 50 từ."
        )
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return "Xin lỗi, đã xảy ra lỗi khi truy vấn mô hình."


def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='vi')
    tts.save(filename)


@app.route('/getReplyAudio')
def get_reply_audio():
    max_wait = 10  # seconds
    poll_interval = 0.5
    waited = 0.0

    print(f">> Client requested {RESPONSE_WAV}")
    while not os.path.exists(RESPONSE_WAV) and waited < max_wait:
        print(f"  Waiting... ({waited:.1f}s/{max_wait}s)")
        time.sleep(poll_interval)
        waited += poll_interval

    if not os.path.exists(RESPONSE_WAV):
        print(">> Error: response.wav not found after waiting.")
        return jsonify({"error": "Audio file not ready"}), 404

    print(">> Streaming response.wav...")

    def generate():
        with open(RESPONSE_WAV, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk
        print(">> Completed playback stream.")
        yield b""  # explicitly signal EOF

    response = Response(generate(), mimetype="audio/wav")

    @response.call_on_close
    def cleanup():
        print(">> Stream closed, cleaning up temporary files...")
        for fpath in [RESPONSE_WAV, RESPONSE_MP3, WAV_FILE]:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    print(f">> Removed: {fpath}")
                except Exception as e:
                    print(f">> Error removing {fpath}: {e}")

    return response



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f'Listening at port {port}')
    app.run(host='0.0.0.0', port=port, threaded=True)
