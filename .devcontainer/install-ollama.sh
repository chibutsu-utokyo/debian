#!/bin/bash
# Install Ollama

if command -v ollama >/dev/null 2>&1; then
  echo "ollama is already installed"
  exit 0
fi

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
echo "ollama installed successfully"
