#!/usr/bin/env bash
set -euo pipefail
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cp main.pdf MMALS_Goal_Adaptive_Geo_RL_FB_article.pdf
