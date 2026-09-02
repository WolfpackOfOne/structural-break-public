#!/usr/bin/env bash
# RT-1320 Phase 1 Stage 2 — alt2 + alt3.
#
# Mirrors EXACTLY the alt1 Stage 1 invocations recorded in
# research/reports/rt1320_promotion/PHASE1_ALT1_STAGE1.md §Provenance:
#
#   catboost   catboost_specialist_2026.py --partition altK --train-only CAT-413 CAT-300
#   teachers   wave7_teacher_nested.py --partition altK --inner-teacher-pair F G   (10 pairs -> 20 vectors)
#   student    armc_residual_student.py --train-outer F --folds-path ... --oof-suffix .altK, then --merge-analyze
#   endpoint   armc_e2_e1_addition_contract.py --oof-suffix .altK
#
# The preregistration (research/RT1320_PROMOTION_PLAN.md §4, §9) is authoritative.
# This script only sequences the runs; it decides nothing. Thresholds were frozen
# before alt1 was run and are NOT evaluated here -- the endpoint JSONs are the
# record, and the verdict is read from them afterwards against §4.
#
# Threads: 8. Phase 0.7 Lever 2 established predictions are BITWISE EQUAL across
# thread counts (max|diff| 0.000e+00); only the model header line differs. So this
# does not confound the partition comparison.
#
# State on entry: alt2 has its 20 nested teachers already; alt3 has none.
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

# wave7_teacher_nested drives sbr.pipeline directly, and pipeline.py binds its
# ROOT/STORE/FEATURES from SBR_ROOT *at import*, defaulting to /home/claude/sb.
# The catboost and student steps take explicit path args so they were unaffected,
# which is why this only surfaced at the first teacher fit.
export SBR_ROOT="$PWD"
export SBR_STORE="$PWD/cache/store"
export SBR_FEATURES="$PWD/cache/features"

THREADS="${THREADS:-8}"
LOGDIR="${LOGDIR:-$HOME/rt1320_stage2_logs}"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { log "FAILED: $*"; log "stopping -- later steps would consume bad inputs"; exit 1; }

run_partition() {
  local K="$1" need_teachers="$2"
  # NOT research/folds/folds_${K}.parquet -- that is a 2-column [id, fold]
  # reassignment table over the 8,000 dev series only, and build_row_arrays
  # rejects it ("ids must be contiguous"). wave2_lib.alt_folds materialises the
  # usable full-length table here: 10,000 rows, lockbox left at -1.
  local FOLDS="cache/_folds_${K}.parquet"
  local SUF=".${K}"
  log "################ partition ${K} ################"

  if [[ "$need_teachers" == "yes" ]]; then
    log "[$K] nested teachers: 10 pair fits -> 20 vectors"
    while read -r _flag F G; do
      [[ -z "${G:-}" ]] && continue
      log "[$K]   inner-teacher-pair $F $G"
      python3 research/scripts/wave7_teacher_nested.py \
        --partition "$K" --inner-teacher-pair "$F" "$G" \
        >> "$LOGDIR/${K}_teachers.log" 2>&1 || die "$K teacher pair $F $G"
    done < <(python3 research/scripts/wave7_teacher_nested.py --list-inner-pairs 2>/dev/null)
  else
    log "[$K] nested teachers already present -- skipping"
  fi

  # Plan §1.1: the sentinel must pass on the partition BEFORE any score is read.
  log "[$K] fold-purity sentinel"
  python3 research/scripts/wave7_teacher_nested.py --partition "$K" --fold-purity-test \
    > "$LOGDIR/${K}_purity.log" 2>&1 || die "$K fold-purity sentinel"
  grep -q "FOLD_PURITY_TEST = PASS" "$LOGDIR/${K}_purity.log" \
    || die "$K fold-purity sentinel did not report PASS"
  log "[$K]   PASS"

  [[ -f "$FOLDS" ]] || die "$K: $FOLDS not materialised (alt_folds writes it during a --partition run)"

  # Plan §1.3: required for the champion lane. Without these there is no E0.
  log "[$K] CatBoost members CAT-413 (RT-1254) + CAT-300 (RT-1255)  ~63 min"
  # --artifact-root is essential: catboost's OOF_DIR is <artifact-root>/research/oof
  # and its DEFAULT root is structural-break-learner-diversity-2026, NOT this
  # worktree. Without this, alt vectors land where the endpoint will not find
  # them (exactly how the first alt2 attempt failed).
  python3 research/scripts/catboost_specialist_2026.py \
    --partition "$K" --train-only CAT-413 CAT-300 --artifact-root "$PWD" \
    > "$LOGDIR/${K}_catboost.log" 2>&1 || die "$K catboost members"
  for m in RT-1254 RT-1255; do
    [[ -f "research/oof/${m}${SUF}.npy" ]] || die "$K: ${m}${SUF}.npy not written to research/oof"
  done

  # --out-dir is mandatory for a partition run: armc_residual_student REFUSES to
  # write alt results into the canonical (tracked) report directory. Mirrors the
  # alt1 convention on disk, research/reports/armc_residual_student.alt1.
  local SDIR="research/reports/armc_residual_student${SUF}"
  log "[$K] student: 5 outer fits -> $SDIR  ~29 min"
  for F in 0 1 2 3 4; do
    log "[$K]   train-outer $F"
    python3 research/scripts/armc_residual_student.py \
      --train-outer "$F" --folds-path "$FOLDS" --oof-suffix "$SUF" \
      --out-dir "$SDIR" --num-threads "$THREADS" \
      >> "$LOGDIR/${K}_student.log" 2>&1 || die "$K student outer $F"
  done
  log "[$K] student merge"
  python3 research/scripts/armc_residual_student.py \
    --merge-analyze --folds-path "$FOLDS" --oof-suffix "$SUF" --out-dir "$SDIR" \
    >> "$LOGDIR/${K}_student.log" 2>&1 || die "$K student merge"

  # --student-dir is required; --catboost-oof-dir must be overridden because the
  # default (structural-break-learner-diversity-2026) holds no alt vectors.
  log "[$K] endpoint: addition contract"
  python3 research/scripts/armc_e2_e1_addition_contract.py \
    --oof-suffix "$SUF" --folds-path "$FOLDS" \
    --student-dir "$SDIR" --catboost-oof-dir research/oof \
    > "$LOGDIR/${K}_endpoint.log" 2>&1 || die "$K endpoint"
  log "[$K] DONE"
}

log "RT-1320 Phase 1 Stage 2 starting. threads=$THREADS logs=$LOGDIR"
log "alt2 first (teachers already present), then alt3 (teachers needed)"
# PARTITIONS lets a partial re-run target what is actually outstanding.
for K in ${PARTITIONS:-alt2 alt3}; do
  case "$K" in
    alt2) run_partition alt2 no ;;
    alt3) run_partition alt3 yes ;;
    *) die "unknown partition $K" ;;
  esac
done

log "################ Stage 2 compute complete ################"
log "Endpoint records written. Read the verdict against RT1320_PROMOTION_PLAN.md §4:"
log "  PASS mean E2-E1 over 4 partitions >= 0.0011 AND >0 on >=3 of 4 AND none worse than -0.0011"
log "  KILL mean below noise floor, OR two or more partitions negative"
log "  INCONCLUSIVE otherwise"
log "Report ALL partitions whatever the outcome (plan anti-gaming clause)."
