#!/usr/bin/env bash
set -euo pipefail

echo "## MATLAB audit"
if command -v matlab >/dev/null 2>&1; then
  echo "PATH matlab: $(command -v matlab)"
else
  echo "PATH matlab: not found"
fi

for candidate in \
  "$HOME/Applications/MATLAB/R2024a/bin/matlab" \
  "$HOME/Applications/MATLAB/R2023b/bin/matlab" \
  "/usr/local/MATLAB/R2024a/bin/matlab" \
  "/usr/local/MATLAB/R2023b/bin/matlab" \
  "/opt/MATLAB/R2024a/bin/matlab" \
  "/opt/MATLAB/R2023b/bin/matlab"; do
  if [ -x "$candidate" ]; then
    echo "candidate: $candidate"
    "$candidate" -batch "disp(version); disp(license('test','communication_toolbox'));" || true
    exit 0
  fi
done

echo "No executable MATLAB candidate found."
exit 1
