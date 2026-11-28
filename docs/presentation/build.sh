#!/bin/bash
# Render the Marp deck to PDF and HTML.
#
# Install marp-cli once:
#   brew install marp-cli
#
# Then from the repository root:
#   bash docs/presentation/build.sh
#
# Outputs:
#   docs/presentation/deck.pdf   — for submission/print
#   docs/presentation/deck.html  — for projecting in a browser
#
# Flags:
#   --theme-set loads theme.css (editorial-light aesthetic)
#   --allow-local-files lets the renderer embed the referenced PNGs
#   --pdf / --html select the output format
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v marp >/dev/null 2>&1; then
  echo "marp-cli not found. Install with: brew install marp-cli" >&2
  exit 1
fi

marp deck.md --theme-set theme.css --allow-local-files --pdf --output deck.pdf
marp deck.md --theme-set theme.css --allow-local-files --html --output deck.html

echo "built deck.pdf and deck.html in docs/presentation/"
