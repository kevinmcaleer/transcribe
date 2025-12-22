#!/bin/bash
# Installation Script for Speech-to-Text Transcription Project
# ==============================================================

set -e

echo "=========================================="
echo "  Speech-to-Text Installation Script"
echo "=========================================="
echo ""

# Detect operating system
OS="$(uname -s)"

echo "Detected OS: $OS"
echo ""

# Install system dependencies
echo "[1/3] Installing system dependencies..."

case "$OS" in
    Linux*)
        echo "Installing PortAudio for Linux..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y portaudio19-dev python3-pyaudio
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y portaudio-devel
        elif command -v pacman &> /dev/null; then
            sudo pacman -S --noconfirm portaudio
        else
            echo "Warning: Could not detect package manager. Please install portaudio manually."
        fi
        ;;
    Darwin*)
        echo "Installing PortAudio for macOS..."
        if command -v brew &> /dev/null; then
            brew install portaudio
        else
            echo "Error: Homebrew not found. Please install Homebrew first:"
            echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "Windows detected. PortAudio should install automatically with PyAudio."
        ;;
    *)
        echo "Unknown OS: $OS"
        echo "Please install PortAudio manually for your system."
        ;;
esac

echo ""
echo "[2/3] Creating virtual environment (optional but recommended)..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""

# Activate virtual environment for pip install
source venv/bin/activate 2>/dev/null || true

echo "[3/3] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment (if not already):"
echo "       source venv/bin/activate"
echo "  2. Run the script:"
echo "       python3 speech_to_text.py"
echo ""
echo "Optional: For offline transcription, install additional packages:"
echo "  pip install openai-whisper    # Whisper (high accuracy, offline)"
echo "  pip install pocketsphinx      # Sphinx (lightweight, offline)"
echo ""
