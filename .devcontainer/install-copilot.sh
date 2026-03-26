#!/bin/bash
# Install GitHub Copilot CLI

if command -v copilot >/dev/null 2>&1; then
  echo "copilot is already installed"
  exit 0
fi

echo "Installing GitHub Copilot CLI..."
curl -fsSL https://gh.io/copilot-install | bash
echo "copilot installed successfully"
