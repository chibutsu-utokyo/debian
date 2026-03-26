#!/bin/bash
# Install Claude Code

if command -v claude >/dev/null 2>&1; then
  echo "claude is already installed"
  exit 0
fi

echo "Installing Claude Code..."
curl -fsSL https://claude.ai/install.sh | bash
echo "claude installed successfully"
