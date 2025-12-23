#!/usr/bin/env python3
"""
Speech to Text Web App using faster-whisper (offline).
ChatGPT-like web interface with start/stop and save functionality.
"""

import threading
import json
import numpy as np
import pyaudio
from flask import Flask, render_template_string, jsonify, request, Response
from faster_whisper import WhisperModel

app = Flask(__name__)

# Global state
state = {
    "is_recording": False,
    "model": None,
    "model_loaded": False,
    "transcript": [],
    "devices": [],
    "selected_device": None,
}

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 300  # Lower = more sensitive to quiet speech
MAX_SILENCE_CHUNKS = 12  # ~3 seconds of silence before transcribing

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech to Text</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #343541;
            color: #ffffff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 2rem;
        }
        .status {
            text-align: center;
            color: #8e8ea0;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }
        .transcript-box {
            flex: 1;
            background-color: #444654;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            min-height: 300px;
            max-height: 500px;
            overflow-y: auto;
            font-size: 1.1rem;
            line-height: 1.6;
        }
        .transcript-box p {
            margin-bottom: 15px;
            padding: 10px 15px;
            background-color: #3e3f4b;
            border-radius: 8px;
        }
        .transcript-box:empty::before {
            content: "Transcribed text will appear here...";
            color: #8e8ea0;
        }
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        select {
            flex: 1;
            min-width: 200px;
            padding: 12px 15px;
            border-radius: 8px;
            border: none;
            background-color: #40414f;
            color: #ffffff;
            font-size: 1rem;
            cursor: pointer;
        }
        select:focus {
            outline: 2px solid #10a37f;
        }
        button {
            padding: 12px 25px;
            border-radius: 8px;
            border: none;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-record {
            background-color: #10a37f;
            color: white;
        }
        .btn-record:hover:not(:disabled) {
            background-color: #0d8a6a;
        }
        .btn-record.recording {
            background-color: #ef4444;
        }
        .btn-record.recording:hover:not(:disabled) {
            background-color: #dc2626;
        }
        .btn-secondary {
            background-color: #565869;
            color: white;
        }
        .btn-secondary:hover:not(:disabled) {
            background-color: #4a4b5c;
        }
        .recording-indicator {
            display: none;
            align-items: center;
            gap: 8px;
            color: #ef4444;
            font-weight: 500;
        }
        .recording-indicator.active {
            display: flex;
        }
        .pulse {
            width: 12px;
            height: 12px;
            background-color: #ef4444;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.1); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speech to Text</h1>
        <div class="status" id="status">Loading model...</div>

        <div class="transcript-box" id="transcript"></div>

        <div class="controls">
            <select id="device-select" disabled>
                <option value="">Loading devices...</option>
            </select>
            <button class="btn-record" id="record-btn" disabled onclick="toggleRecording()">
                Start Recording
            </button>
            <button class="btn-secondary" onclick="saveTranscript()">Save</button>
            <button class="btn-secondary" onclick="clearTranscript()">Clear</button>
            <button class="btn-secondary" onclick="refreshDevices()">Refresh Devices</button>
            <div class="recording-indicator" id="recording-indicator">
                <div class="pulse"></div>
                Recording
            </div>
        </div>
    </div>

    <script>
        let isRecording = false;
        let pollInterval = null;

        async function init() {
            await loadDevices();
            await checkModelStatus();
        }

        async function loadDevices() {
            const res = await fetch('/devices');
            const data = await res.json();
            const select = document.getElementById('device-select');
            select.innerHTML = data.devices.map((d, i) =>
                `<option value="${d.index}">[${d.index}] ${d.name}</option>`
            ).join('');
            select.disabled = false;
        }

        async function checkModelStatus() {
            const res = await fetch('/status');
            const data = await res.json();
            document.getElementById('status').textContent = data.status;
            if (data.model_loaded) {
                document.getElementById('record-btn').disabled = false;
            } else {
                setTimeout(checkModelStatus, 1000);
            }
        }

        async function toggleRecording() {
            const btn = document.getElementById('record-btn');
            const indicator = document.getElementById('recording-indicator');
            const select = document.getElementById('device-select');

            if (!isRecording) {
                const device = select.value;
                const res = await fetch('/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({device: parseInt(device)})
                });
                if (res.ok) {
                    isRecording = true;
                    btn.textContent = 'Stop Recording';
                    btn.classList.add('recording');
                    indicator.classList.add('active');
                    select.disabled = true;
                    document.getElementById('status').textContent = 'Recording... Speak now';
                    pollInterval = setInterval(pollTranscript, 500);
                }
            } else {
                await fetch('/stop', {method: 'POST'});
                isRecording = false;
                btn.textContent = 'Start Recording';
                btn.classList.remove('recording');
                indicator.classList.remove('active');
                select.disabled = false;
                document.getElementById('status').textContent = 'Stopped';
                clearInterval(pollInterval);
                await pollTranscript();
            }
        }

        async function pollTranscript() {
            const res = await fetch('/transcript');
            const data = await res.json();
            const box = document.getElementById('transcript');
            box.innerHTML = data.transcript.map(t => `<p>${t}</p>`).join('');
            box.scrollTop = box.scrollHeight;
        }

        function saveTranscript() {
            const box = document.getElementById('transcript');
            const text = Array.from(box.querySelectorAll('p')).map(p => p.textContent).join('\\n\\n');
            if (!text) {
                alert('No transcript to save');
                return;
            }
            const now = new Date();
            const timestamp = now.getFullYear() +
                String(now.getMonth() + 1).padStart(2, '0') +
                String(now.getDate()).padStart(2, '0') + '_' +
                String(now.getHours()).padStart(2, '0') +
                String(now.getMinutes()).padStart(2, '0') +
                String(now.getSeconds()).padStart(2, '0');
            const blob = new Blob([text], {type: 'text/plain'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `transcript_${timestamp}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }

        async function clearTranscript() {
            await fetch('/clear', {method: 'POST'});
            document.getElementById('transcript').innerHTML = '';
        }

        async function refreshDevices() {
            const select = document.getElementById('device-select');
            select.disabled = true;
            select.innerHTML = '<option value="">Scanning...</option>';
            await loadDevices();
            document.getElementById('status').textContent = 'Devices refreshed';
        }

        init();
    </script>
</body>
</html>
"""


def load_model():
    """Load the Whisper model."""
    print("Loading Whisper model...")
    state["model"] = WhisperModel("medium", device="cpu", compute_type="int8")
    state["model_loaded"] = True
    print("Model loaded!")


def get_devices():
    """Get list of input devices."""
    p = pyaudio.PyAudio()
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            devices.append({"index": i, "name": info['name']})
    p.terminate()
    return devices


def record_loop(device_idx):
    """Main recording loop."""
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_idx,
        frames_per_buffer=4000
    )

    audio_buffer = []
    silence_chunks = 0

    try:
        while state["is_recording"]:
            data = stream.read(4000, exception_on_overflow=False)
            data = np.frombuffer(data, dtype=np.int16)
            audio_buffer.append(data)

            peak = np.max(np.abs(data))
            if peak < SILENCE_THRESHOLD:
                silence_chunks += 1
            else:
                silence_chunks = 0

            buffer_duration = len(audio_buffer) * 4000 / SAMPLE_RATE
            if (silence_chunks >= MAX_SILENCE_CHUNKS and buffer_duration > 2.0) or buffer_duration > 30:
                if len(audio_buffer) > 0:
                    audio_data = np.concatenate(audio_buffer).flatten().astype(np.float32) / 32768.0
                    segments, _ = state["model"].transcribe(audio_data, beam_size=5, language="en")
                    text = " ".join([seg.text for seg in segments]).strip()
                    if text:
                        state["transcript"].append(text)
                    audio_buffer = []
                    silence_chunks = 0
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/devices')
def devices():
    return jsonify({"devices": get_devices()})


@app.route('/status')
def status():
    if state["model_loaded"]:
        return jsonify({"status": "Ready - Select a device and click Start", "model_loaded": True})
    return jsonify({"status": "Loading model...", "model_loaded": False})


@app.route('/start', methods=['POST'])
def start():
    if state["is_recording"]:
        return jsonify({"error": "Already recording"}), 400

    data = request.json
    device_idx = data.get('device', 0)

    state["is_recording"] = True
    state["selected_device"] = device_idx

    thread = threading.Thread(target=record_loop, args=(device_idx,), daemon=True)
    thread.start()

    return jsonify({"status": "Recording started"})


@app.route('/stop', methods=['POST'])
def stop():
    state["is_recording"] = False
    return jsonify({"status": "Recording stopped"})


@app.route('/transcript')
def transcript():
    return jsonify({"transcript": state["transcript"]})


@app.route('/clear', methods=['POST'])
def clear():
    state["transcript"] = []
    return jsonify({"status": "Cleared"})


if __name__ == '__main__':
    # Load model in background thread
    threading.Thread(target=load_model, daemon=True).start()

    print("\n" + "=" * 50)
    print("Speech to Text Web App")
    print("=" * 50)
    print("\nOpen http://localhost:5000 in your browser\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
