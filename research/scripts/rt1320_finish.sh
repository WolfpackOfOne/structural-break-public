#!/usr/bin/env bash
# Chain the RT-1320 finishing sequence behind the local fit.
#
# Waits for the fit to produce both deliverables, then:
#   1. assembles the 8-member artifact          (clears registry item 4)
#   2. runs the artifact causality gate         (clears registry item 2)
#   3. writes the deployability audit
#
# Nothing here trains. It is safe to leave running overnight: if the fit never
# finishes it times out and exits non-zero without touching anything.
#
#   nohup research/scripts/rt1320_finish.sh > ~/rt1320_finish.log 2>&1 &
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIT_DIR="${FIT_DIR:-$HOME/rt1320_local_fit/rt1320_final10k_fit}"
OUT_DIR="${OUT_DIR:-$REPO/models/rt1320_final}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-12}"
POLL_SECONDS="${POLL_SECONDS:-120}"

MODEL="$FIT_DIR/model.txt.7"
CAL="$FIT_DIR/RT-1320_student_scdf.json"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "repo    : $REPO"
log "fit dir : $FIT_DIR"
log "out dir : $OUT_DIR"
log "waiting for the fit to produce both deliverables (timeout ${TIMEOUT_HOURS}h)"

deadline=$(( $(date +%s) + TIMEOUT_HOURS * 3600 ))
while :; do
  if [[ -f "$MODEL" && -f "$CAL" ]]; then
    # Guard against reading a file mid-write: require the size to hold steady.
    a=$(wc -c < "$MODEL"); sleep 20; b=$(wc -c < "$MODEL")
    if [[ "$a" == "$b" && "$a" -gt 0 ]]; then
      log "both deliverables present and stable (model.txt.7 = $a bytes)"
      break
    fi
    log "model.txt.7 still growing ($a -> $b); waiting"
  fi
  if (( $(date +%s) > deadline )); then
    log "TIMEOUT after ${TIMEOUT_HOURS}h. Still missing:"
    [[ -f "$MODEL" ]] || log "  model.txt.7            (--stage final)"
    [[ -f "$CAL"   ]] || log "  RT-1320_student_scdf.json (--stage calib)"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

cd "$REPO" || exit 1

log "=== 1/3 assembling the 8-member artifact ==="
if ! python3 research/scripts/rt1320_assemble_artifact.py \
      --fit-dir "$FIT_DIR" --out "$OUT_DIR" --force; then
  log "ASSEMBLY FAILED -- stopping before the gates."
  exit 1
fi

log "=== 2/3 artifact causality gate ==="
python3 research/scripts/rt1320_promotion_prep.py validate-artifact \
    --model-dir "$OUT_DIR" || log "causality gate reported a failure (see above)"

log "=== 3/3 deployability audit ==="
python3 research/scripts/rt1320_promotion_prep.py audit \
    --model-dir "$OUT_DIR" || log "audit reported a failure (see above)"

log "=== done ==="
log "artifact : $OUT_DIR"
log "audit    : $REPO/engineering/reports/rt1320_promotion_prep/"
for f in model.txt.7 RT-1320_student_scdf.json manifest.json; do
  [[ -f "$OUT_DIR/$f" ]] && log "  $(shasum -a 256 "$OUT_DIR/$f" | awk '{print $1}')  $f"
done
