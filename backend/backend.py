# --- START OF THE DEFINITIVE FIX ---
# This MUST be the first piece of code to run to prevent library conflicts.
# It patches Python's standard libraries to be compatible with the database driver's networking.
from gevent import monkey
monkey.patch_all()
# --- END OF THE DEFINITIVE FIX ---

from flask import Flask, request, jsonify
import os
import json
import librosa
import numpy as np
import traceback
from dtw import dtw
from gradio_client import Client, handle_file
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

from cassandra.io.geventreactor import GeventConnection

#from gevent import monkey
#monkey.patch_all()

# --- CONFIGURATION ---
ASTRA_BUNDLE_PATH = "secure-connect-vocal-analyser.zip" 
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN") # Replace with your Astra token
ASTRA_KEYSPACE = "mykeyspace" 

app = Flask(__name__)

# --- CORE LOGIC FUNCTIONS (same as before) ---
def separate_vocals(audio_path: str):
    client = Client("r3gm/Audio_separator")
    result = client.predict(media_file=handle_file(audio_path), api_name="/sound_separate")
    if result and isinstance(result, list) and len(result) > 0: return result[0]
    return None

def extract_features(audio_path: str):
    y, sr = librosa.load(audio_path, sr=22050)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0[~voiced_flag] = 0
    return f0

def get_original_features(song_id: str):
    cloud_config = {'secure_connect_bundle': ASTRA_BUNDLE_PATH}
    auth_provider = PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider, connection_class=GeventConnection)
    session = cluster.connect()
    session.set_keyspace(ASTRA_KEYSPACE)
    row = session.execute("SELECT features_json FROM vocal_features WHERE song_id = %s", (song_id,)).one()
    cluster.shutdown()
    if row: return json.loads(row.features_json)
    return None

# --- THE API ENDPOINT ---
@app.route('/analyse', methods=['POST'])
def analyse_endpoint():
    if 'user_audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    user_audio_file = request.files['user_audio']
    song_id = request.form.get('song_id')

    # Save the uploaded file temporarily
    temp_filename = "temp_user_audio.wav"
    user_audio_file.save(temp_filename)

    try:
        original_features = get_original_features(song_id)
        if not original_features: raise ValueError("Original song not found")
        original_pitch = np.array(original_features['pitch_contour_hz'])

        user_vocals_path = separate_vocals(temp_filename)
        if not user_vocals_path: raise ValueError("Vocal separation failed")

        user_pitch = extract_features(user_vocals_path)

        original_pitch_norm = (original_pitch - np.mean(original_pitch)) / (np.std(original_pitch) + 1e-6)
        user_pitch_norm = (user_pitch - np.mean(user_pitch)) / (np.std(user_pitch) + 1e-6)

        alignment = dtw(original_pitch_norm, user_pitch_norm, keep_internals=True)
        pitch_score = max(0, 100 * (1 - (alignment.distance / len(original_pitch_norm))))

        return jsonify({"score": round(pitch_score, 2)})

    except Exception as e:
        print("--- DETAILED ERROR TRACEBACK ---")
        traceback.print_exc()
        print("------------------------------------")
        return jsonify({"error": str(e)}), 500
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == '__main__':
    app.run(port=5000)