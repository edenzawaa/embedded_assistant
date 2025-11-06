from flask import Flask, request, jsonify, Response
import os, time, threading
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as genai
from dotenv import load_dotenv

app = Flask(__name__)

WAV_FILE = 'recording.wav'
RESPONSE_MP3 = 'response.mp3'
RESPONSE_WAV = 'response.wav'

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# --- GLOBAL STATUS FLAGS ---
status = {"processing": False, "ready": False, "error": None}

# ----------------------------


@app.route("/uploadAudio", methods=["POST"])
def upload_audio():
    global status
    try:
        file = request.files["audio"]
        file.save(WAV_FILE)
        print(">> Audio received:", WAV_FILE)

        # Reset flags before starting a new thread
        status = {"processing": True, "ready": False, "error": None}

        # Process in a background thread
        threading.Thread(target=process_audio, daemon=True).start()

        return jsonify({"status": "processing"}), 200

    except Exception as e:
        status = {"processing": False, "ready": False, "error": str(e)}
        return jsonify({"error": str(e)}), 500


@app.route("/checkStatus", methods=["GET"])
def check_status():
    """ESP32 calls this periodically to check if reply is ready."""
    return jsonify(status)


@app.route("/getReplyAudio", methods=["GET"])
def get_reply_audio():
    """ESP32 calls this once /checkStatus says ready=True."""
    global status
    try:
        if not os.path.exists(RESPONSE_MP3):
            return jsonify({"error": "No audio ready"}), 404

        def generate():
            with open(RESPONSE_MP3, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
            print(">> Completed playback stream.")

        response = Response(generate(), mimetype="audio/mpeg")

        @response.call_on_close
        def cleanup():
            for fpath in [RESPONSE_MP3, WAV_FILE]:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f">> Removed: {fpath}")
            status = {"processing": False, "ready": False, "error": None}

        return response

    except Exception as e:
        status = {"processing": False, "ready": False, "error": str(e)}
        return jsonify({"error": str(e)}), 500


def process_audio():
    global status
    try:
        print(">> Sending audio to Gemini...")
        with open(WAV_FILE, "rb") as f:
            audio_data = f.read()

        # Directly transcribe + respond
        result = model.generate_content([
            {"mime_type": "audio/wav", "data": audio_data},
            {"text": "Nghe nội dung âm thanh, hiểu người dùng nói gì và phản hồi ngắn gọn bằng tiếng Việt."}
        ])

        reply_text = result.text.strip() if result.text else "Xin lỗi, tôi không nghe rõ."
        print(f">> Gemini reply: {reply_text}")

        # Generate TTS
        print(">> Generating voice reply...")
        gTTS(reply_text, lang="vi").save(RESPONSE_MP3)

        status = {"processing": False, "ready": True, "error": None}
        print(">> Processing complete. Audio ready.")

    except Exception as e:
        status = {"processing": False, "ready": False, "error": str(e)}
        print(">> ERROR in process_audio:", e)
