#!/bin/bash
# ═══════════════════════════════════════════
#  FootTrack — Setup & Run Script
# ═══════════════════════════════════════════

set -e
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RESET="\033[0m"

echo -e "${BOLD}${GREEN}⚽  FootTrack — AI Football Retracker${RESET}"
echo -e "────────────────────────────────────────"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3 not found. Please install Python 3.9+${RESET}"
    exit 1
fi

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${YELLOW}ffmpeg not found.${RESET}"
    echo "Install it with:"
    echo "  macOS:   brew install ffmpeg"
    echo "  Ubuntu:  sudo apt install ffmpeg"
    echo "  Windows: https://ffmpeg.org/download.html"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BACKEND_DIR="$SCRIPT_DIR/backend"

# Create virtual environment if needed
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "\n${BOLD}Creating virtual environment...${RESET}"
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

# Activate venv
source "$SCRIPT_DIR/.venv/bin/activate"

# Install dependencies
echo -e "\n${BOLD}Installing dependencies...${RESET}"
pip install --upgrade pip -q
pip install -r "$BACKEND_DIR/requirements.txt" -q

echo -e "\n${GREEN}✓ Dependencies installed${RESET}"

# Start backend
echo -e "\n${BOLD}Starting backend on http://localhost:5000${RESET}"
echo -e "Open ${BOLD}frontend/index.html${RESET} in your browser\n"
echo -e "────────────────────────────────────────"

cd "$BACKEND_DIR"
python3 app.py
