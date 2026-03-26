#!/bin/bash
# Install OpenCode CLI

if command -v opencode >/dev/null 2>&1; then
  echo "opencode is already installed"
  exit 0
fi

echo "Installing OpenCode..."
curl -fsSL https://opencode.ai/install | bash
echo "opencode installed successfully"
