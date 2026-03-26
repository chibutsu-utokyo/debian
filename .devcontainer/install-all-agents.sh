#!/bin/bash
# Install all AI coding assistants

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing all AI coding assistants..."
echo ""

"$SCRIPT_DIR/install-copilot.sh"
"$SCRIPT_DIR/install-opencode.sh"
"$SCRIPT_DIR/install-codex.sh"
"$SCRIPT_DIR/install-claude.sh"
"$SCRIPT_DIR/install-ollama.sh"

echo ""
echo "All AI agents installed successfully!"
