#!/usr/bin/env python3
"""
Speech to Text using faster-whisper (offline).
Transcribes microphone audio and outputs to screen and output.txt
"""

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent / "output.txt"
SAMPLE_RATE = 16000


def select_input_device():
    """List input devices and let user choose."""
    p = pyaudio.PyAudio()
    print("\nAvailable input devices:")
    print("-" * 40)
    input_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxInputChannels'] > 0:
            print(f"  [{i}] {info['name']}")
            input_devices.append(i)
    print("-" * 40)
    p.terminate()

    choice = input("Enter device number: ").strip()
    try:
        idx = int(choice)
        if idx in input_devices:
            return idx
    except ValueError:
        pass
    print("Invalid choice, using default")
    return None


def main():
    print("\nSpeech to Text (faster-whisper)")
    print("=" * 40)

    # Select input device
    device_idx = select_input_device()

    # Load model (downloads on first run)
    print("\nLoading model (first run will download ~1.5GB)...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("Model ready!")

    # Clear output file
    OUTPUT_FILE.write_text("")

    print("\nListening... Press Ctrl+C to stop.")
    print("(Transcription happens after pauses in speech)\n")
    print("-" * 40)

    # Setup PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_idx,
        frames_per_buffer=4000
    )

    # Buffer to accumulate audio
    audio_buffer = []
    silence_threshold = 500
    silence_chunks = 0
    max_silence_chunks = 6

    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            data = np.frombuffer(data, dtype=np.int16)
            audio_buffer.append(data)

            # Check for silence
            peak = np.max(np.abs(data))
            if peak < silence_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

            # Transcribe after silence or buffer gets large
            buffer_duration = len(audio_buffer) * 4000 / SAMPLE_RATE
            if (silence_chunks >= max_silence_chunks and buffer_duration > 0.5) or buffer_duration > 30:
                if len(audio_buffer) > 0:
                    # Combine buffer and convert to float32
                    audio_data = np.concatenate(audio_buffer).flatten().astype(np.float32) / 32768.0

                    # Transcribe
                    segments, _ = model.transcribe(audio_data, beam_size=5, language="en")
                    text = " ".join([seg.text for seg in segments]).strip()

                    if text:
                        print(f"\r{text}")
                        with open(OUTPUT_FILE, "a") as f:
                            f.write(text + "\n")

                    # Reset buffer
                    audio_buffer = []
                    silence_chunks = 0

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print(f"Transcript saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
