#!/usr/bin/env bash
# Trackable perf protocol: run the bench, append a logbook entry + CSV row, optionally tag.
# See docs/PERF-LOGBOOK.md for conventions.
#
# Usage:
#   scripts/perf-log.sh <id> "<title>" [--variants v1,v2] [--tokens N] [--ctx N] [--prompt P]
#                                      [--local] [--tag] [--no-run --rows "var,tok_s,rss;..."]
#   <id>      short slug -> tag perf/<id> (e.g. a1-packed-llama)
#   --local   add -PuseLocalSkainet=true (composite build vs ../SKaiNET-transformers source)
#   --tag     create annotated git tag perf/<id> with the note
#   --no-run  skip the bench; use --rows "variant,tok_s,rss;variant,tok_s,rss"
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

id="${1:?usage: perf-log.sh <id> \"<title>\" [opts]}"; title="${2:?title required}"; shift 2
variants="python-baseline,eager-jvm"; tokens=16; ctx=256; prompt="What is quantization?"
local_flag=""; do_tag=0; run=1; rows=""
while [ $# -gt 0 ]; do case "$1" in
  --variants) variants="$2"; shift 2;;
  --tokens) tokens="$2"; shift 2;;
  --ctx) ctx="$2"; shift 2;;
  --prompt) prompt="$2"; shift 2;;
  --local) local_flag="-PuseLocalSkainet=true"; shift;;
  --tag) do_tag=1; shift;;
  --no-run) run=0; shift;;
  --rows) rows="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done

logbook="docs/PERF-LOGBOOK.md"; csv="docs/perf-history.csv"
date="$(date +%F)"; sha="$(git rev-parse --short HEAD)"
args="bench --variants $variants --tokens $tokens --ctx $ctx --temperature 0.01 --prompt \"$prompt\""

if [ "$run" = 1 ]; then
  echo "Running: ./gradlew $local_flag :bench:runJvm --args='$args'"
  out="$(./gradlew $local_flag :bench:runJvm --args="$args" -q --console=plain 2>&1 || true)"
  # parse the comparison table: variant tok/s s/tok load infer RSS notes...
  rows="$(printf '%s\n' "$out" | awk 'NF>=6 && $1 ~ /^(python-baseline|eager-jvm|eager-native|iree-cpu|iree-torq)$/ {print $1","$2","$6}' | paste -sd';' -)"
fi
[ -n "$rows" ] || { echo "no metric rows parsed/provided" >&2; exit 1; }
echo "metrics: $rows"

# CSV rows
IFS=';' read -ra rr <<< "$rows"
for r in "${rr[@]}"; do
  v="${r%%,*}"; rest="${r#*,}"; toks="${rest%%,*}"; rssv="${rest#*,}"
  echo "$date,perf/$id,$v,Q4_K_M,$toks,$rssv,,$sha,$title" >> "$csv"
done

# Logbook entry, inserted just under the "# Entries (newest first)" header.
# Write to a temp file and inject via awk getline (portable; BSD awk rejects multi-line -v).
entry_file="$(mktemp)"
{
  echo ""
  echo "### perf/$id — $title  ($date)"
  echo "- What:   $title"
  echo "- How:    @ $sha (fill in mechanism + key files)"
  echo "- Impact: $rows  (vs baseline 0.17 tok/s @ 8070 MB)"
  echo "- Run:    ./gradlew $local_flag :bench:runJvm --args='$args'"
} > "$entry_file"
tmp="$(mktemp)"
awk -v ef="$entry_file" '
  {print}
  /^# Entries \(newest first\)/ { while ((getline line < ef) > 0) print line; close(ef) }
' "$logbook" > "$tmp" && mv "$tmp" "$logbook"
rm -f "$entry_file"
echo "Appended entry to $logbook + rows to $csv. Edit How/Impact as needed."

note="$title — $rows (sha $sha)"
if [ "$do_tag" = 1 ]; then
  git tag -a "perf/$id" -m "$note" && echo "Created tag perf/$id"
else
  echo "To tag: git tag -a perf/$id -m \"$note\""
fi
