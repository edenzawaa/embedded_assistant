from flask import Flask, request, jsonify, Response
import os, threading
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as genai
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

app = Flask(__name__)

WAV_FILE = 'recording.wav'
RESPONSE_WAV = 'response.wav'

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# --- GLOBAL STATUS FLAGS ---
status = {"processing": False, "ready": False, "error": None}


@app.route("/uploadAudio", methods=["POST"])
def upload_audio():
    global status
    try:
        # Use raw data
        data = request.data
        if not data:
            raise ValueError("No data received")
        
        with open(WAV_FILE, "wb") as f:
            f.write(data)
        
        status = {"processing": True, "ready": False, "error": None}
        threading.Thread(target=process_audio, daemon=True).start()
        return jsonify({"status": "processing"}), 200

    except Exception as e:
        status = {"processing": False, "ready": False, "error": str(e)}
        return jsonify({"error": str(e)}), 500



@app.route("/checkStatus", methods=["GET"])
def check_status():
    return jsonify(status)


@app.route("/getReplyAudio", methods=["GET"])
def get_reply_audio():
    global status
    try:
        if not os.path.exists(RESPONSE_WAV):
            return jsonify({"error": "No audio ready"}), 404

        def generate():
            with open(RESPONSE_WAV, "rb") as f:
                while chunk := f.read(8192):
                    yield chunk
            print(">> Completed WAV stream.")

        response = Response(generate(), mimetype="audio/wav")

        @response.call_on_close
        def cleanup():
            for fpath in [RESPONSE_WAV, WAV_FILE]:
                if os.path.exists(fpath):
                    os.remove(fpath)
                    print(f">> Removed: {fpath}")
            status.update({"processing": False, "ready": False, "error": None})

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

        # Gemini transcription + reply
        result = model.generate_content([
            {"mime_type": "audio/wav", "data": audio_data},
            {"text": "Nghe nội dung âm thanh, hiểu người dùng nói gì và phản hồi ngắn gọn bằng tiếng Việt."}
        ])

        reply_text = result.text.strip() if result.text else "Xin lỗi, tôi không nghe rõ."
        print(f">> Gemini reply: {reply_text}")

        # Generate MP3 first, then convert to WAV
        tts_mp3 = "temp_response.mp3"
        gTTS(reply_text, lang="vi").save(tts_mp3)
        AudioSegment.from_mp3(tts_mp3).export(RESPONSE_WAV, format="wav")
        os.remove(tts_mp3)

        status.update({"processing": False, "ready": True, "error": None})
        print(">> Processing complete. WAV ready.")

    except Exception as e:
        status.update({"processing": False, "ready": False, "error": str(e)})
        print(">> ERROR in process_audio:", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Listening at port {port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
