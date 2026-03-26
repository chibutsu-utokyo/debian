#!/bin/bash
# Install Codex CLI

if command -v codex >/dev/null 2>&1; then
  echo "codex is already installed"
  exit 0
fi

echo "Installing Codex..."
curl -fsSL https://github.com/openai/codex/releases/latest/download/install.sh | bash
echo "codex installed successfully"
