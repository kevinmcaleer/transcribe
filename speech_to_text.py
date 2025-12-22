#!/usr/bin/env python3
"""
Speech to Text Application
Listens to audio from microphone and transcribes speech in real-time.

Requirements:
    pip install SpeechRecognition pyaudio

On Linux you may also need:
    sudo apt-get install portaudio19-dev python3-pyaudio

On Mac:
    brew install portaudio
"""

import speech_recognition as sr
import sys


def list_microphones():
    """List all available microphones."""
    print("\nAvailable microphones:")
    for i, mic_name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{i}] {mic_name}")
    print()


def transcribe_continuous(recognizer, microphone, engine="google"):
    """
    Continuously listen and transcribe speech.
    
    Args:
        recognizer: SpeechRecognition Recognizer instance
        microphone: SpeechRecognition Microphone instance
        engine: Recognition engine to use ('google', 'sphinx', 'whisper')
    """
    print(f"\n🎤 Listening... (using {engine})")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")
    print("-" * 50)
    
    while True:
        try:
            with microphone as source:
                # Listen for speech
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=15)
            
            # Transcribe based on selected engine
            try:
                if engine == "google":
                    text = recognizer.recognize_google(audio)
                elif engine == "sphinx":
                    text = recognizer.recognize_sphinx(audio)
                elif engine == "whisper":
                    text = recognizer.recognize_whisper(audio, model="base")
                else:
                    text = recognizer.recognize_google(audio)
                
                print(f"📝 {text}")
                
            except sr.UnknownValueError:
                print("... (couldn't understand)")
            except sr.RequestError as e:
                print(f"⚠️  API error: {e}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Stopped listening.")
            break


def transcribe_once(recognizer, microphone, engine="google"):
    """
    Listen once and transcribe a single phrase.
    
    Returns:
        str: Transcribed text or None if failed
    """
    print("\n🎤 Listening for a single phrase...")
    
    with microphone as source:
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
    
    print("Processing...")
    
    try:
        if engine == "google":
            text = recognizer.recognize_google(audio)
        elif engine == "sphinx":
            text = recognizer.recognize_sphinx(audio)
        elif engine == "whisper":
            text = recognizer.recognize_whisper(audio, model="base")
        else:
            text = recognizer.recognize_google(audio)
        
        return text
        
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"API error: {e}")
        return None


def main():
    # Initialize recognizer
    recognizer = sr.Recognizer()
    
    # Select microphone
    try:
        microphone = sr.Microphone()
    except OSError as e:
        print(f"❌ Error accessing microphone: {e}")
        print("\nMake sure you have a microphone connected and PyAudio installed.")
        print("Install with: pip install pyaudio")
        sys.exit(1)
    
    # Calibrate for ambient noise
    print("🔧 Calibrating for ambient noise... (please wait)")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print("✅ Calibration complete!")
    
    # Menu
    print("\n" + "=" * 50)
    print("Speech to Text App")
    print("=" * 50)
    print("\nRecognition engines available:")
    print("  1. Google (free, requires internet)")
    print("  2. Sphinx (offline, less accurate)")
    print("  3. Whisper (offline, requires openai-whisper)")
    print("\nModes:")
    print("  c - Continuous listening")
    print("  s - Single phrase")
    print("  m - List microphones")
    print("  q - Quit")
    
    engine_map = {"1": "google", "2": "sphinx", "3": "whisper"}
    current_engine = "google"
    
    while True:
        print(f"\nCurrent engine: {current_engine}")
        choice = input("Enter choice (1-3 for engine, c/s/m/q): ").strip().lower()
        
        if choice in engine_map:
            current_engine = engine_map[choice]
            print(f"Switched to {current_engine}")
        elif choice == "c":
            transcribe_continuous(recognizer, microphone, current_engine)
        elif choice == "s":
            result = transcribe_once(recognizer, microphone, current_engine)
            if result:
                print(f"\n📝 Transcribed: {result}")
        elif choice == "m":
            list_microphones()
        elif choice == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
