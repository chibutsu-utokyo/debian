#!/usr/bin/env bash
set -euo pipefail

echo "[install-latex] Installing minimal TeX Live with Japanese support..."

sudo apt-get update
sudo apt-get install -y texlive-latex-recommended texlive-lang-japanese latexmk

echo "[install-latex] Installation finished. You can compile with e.g. 'latexmk -pdf main.tex'." 
