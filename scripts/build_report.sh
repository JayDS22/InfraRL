#!/bin/bash
# Concatenate docs/report/sections/*.md into docs/REPORT.md with the shared
# header. Idempotent; safe to re-run after editing any section file.
set -euo pipefail
cd "$(dirname "$0")/.."

cat > docs/REPORT.md <<'HDR'
# InfraRL: Constrained Reinforcement Learning for LLM Inference Stack Selection

Jay Guwalani, Anupama Sharma, Vishnu Sai Vardhan Reddy Basi Reddy Gari
DATA635, Reinforcement Learning, University of Maryland, Fall 2025
Instructor: Prof. Samet Ayan
Submission: 2026-12-07

---

HDR

count=0
for f in docs/report/sections/*.md; do
  cat "$f" >> docs/REPORT.md
  printf '\n' >> docs/REPORT.md
  count=$((count + 1))
done
echo "rebuilt docs/REPORT.md from $count sections ($(wc -l < docs/REPORT.md | tr -d ' ') lines)"
