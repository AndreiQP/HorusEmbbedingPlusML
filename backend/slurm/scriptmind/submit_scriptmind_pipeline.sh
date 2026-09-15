#!/bin/bash
# Submit the isolated ScriptMind DAG. Run from any checkout location.
set -euo pipefail

DRY_RUN=0
FORCE=0
FROM_STAGE="audit"
ONLY_STAGE=""
WITH_ABLATIONS=0
STAGES=(audit annotate adjudicate build train evaluate report)

usage() {
    echo "Usage: $0 [--dry-run] [--resume] [--force] [--from STAGE] [--only STAGE] [--with-ablations]"
}

stage_index() {
    local wanted="$1"
    local index
    for index in "${!STAGES[@]}"; do
        [[ "${STAGES[$index]}" == "$wanted" ]] && { echo "$index"; return 0; }
    done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --resume) shift ;;
        --force) FORCE=1; shift ;;
        --with-ablations) WITH_ABLATIONS=1; shift ;;
        --from) FROM_STAGE="${2:?--from requires a stage}"; shift 2 ;;
        --only) ONLY_STAGE="${2:?--only requires a stage}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

FROM_INDEX="$(stage_index "$FROM_STAGE")" || { echo "Invalid --from stage: $FROM_STAGE" >&2; exit 2; }
if [[ -n "$ONLY_STAGE" ]]; then
    stage_index "$ONLY_STAGE" >/dev/null || { echo "Invalid --only stage: $ONLY_STAGE" >&2; exit 2; }
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd -P)"
mkdir -p "$PROJECT_ROOT/slurm_logs/scriptmind"
cd "$PROJECT_ROOT"

should_run() {
    local stage="$1"
    [[ -n "$ONLY_STAGE" ]] && [[ "$stage" == "$ONLY_STAGE" ]] && return 0
    [[ -n "$ONLY_STAGE" ]] && return 1
    local index
    index="$(stage_index "$stage")"
    [[ "$index" -ge "$FROM_INDEX" ]]
}

if [[ "$DRY_RUN" == "0" ]] && { should_run annotate || should_run adjudicate; }; then
    : "${GEMINI_API_KEY:?Export GEMINI_API_KEY before submitting Gemini stages}"
fi

EXPORTS="ALL,SCRIPTMIND_FORCE=$FORCE,SCRIPTMIND_NUM_SHARDS=32"
LAST_DEP=""

submit() {
    local description="$1"; shift
    local mail_args=()
    if [[ -n "${SLURM_MAIL_USER:-}" ]]; then
        mail_args+=(--mail-user="$SLURM_MAIL_USER" --mail-type=BEGIN,END,FAIL)
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '[dry-run] %s:' "$description" >&2
        printf ' %q' sbatch --parsable "${mail_args[@]}" "$@" >&2
        printf '\n' >&2
        echo "dry-${description//[^a-zA-Z0-9]/-}"
    else
        local response
        response="$(sbatch --parsable "${mail_args[@]}" "$@")"
        echo "${response%%;*}"
    fi
}

dep_args=()
set_dep() {
    dep_args=()
    [[ -n "${1:-}" ]] && dep_args+=(--dependency="afterok:$1")
}

if should_run audit; then
    LAST_DEP="$(submit audit --export="$EXPORTS" --job-name=scriptmind_audit "$SCRIPT_DIR/run_scriptmind_cpu.sbatch" audit)"
fi

if should_run annotate; then
    set_dep "$LAST_DEP"
    GEMINI_JOB="$(submit annotate-gemini "${dep_args[@]}" --export="$EXPORTS" --job-name=scriptmind_gemini \
        --array=0-31%1 --cpus-per-task=4 --mem=16G --time=1-00:00:00 \
        "$SCRIPT_DIR/run_scriptmind_annotation.sbatch" gemini)"
    LOCAL_JOB="$(submit annotate-local "${dep_args[@]}" --export="$EXPORTS" --job-name=scriptmind_local \
        --array=0-31%2 --gres=gpu:1 \
        "$SCRIPT_DIR/run_scriptmind_annotation.sbatch" local)"
    LAST_DEP="${GEMINI_JOB}:${LOCAL_JOB}"
fi

if should_run adjudicate; then
    set_dep "$LAST_DEP"
    ADJUDICATE_JOB="$(submit adjudicate "${dep_args[@]}" --export="$EXPORTS" --job-name=scriptmind_adjudicate \
        --array=0-31%1 --cpus-per-task=4 --mem=16G --time=1-00:00:00 \
        "$SCRIPT_DIR/run_scriptmind_annotation.sbatch" adjudicate)"
    set_dep "$ADJUDICATE_JOB"
    LAST_DEP="$(submit annotation-quality "${dep_args[@]}" --export="$EXPORTS" --job-name=scriptmind_quality \
        "$SCRIPT_DIR/run_scriptmind_cpu.sbatch" quality)"
fi

if should_run build; then
    set_dep "$LAST_DEP"
    LAST_DEP="$(submit build-csid "${dep_args[@]}" --export="$EXPORTS" --job-name=scriptmind_build \
        "$SCRIPT_DIR/run_scriptmind_cpu.sbatch" build)"
fi

TRAIN_DEPS=""
if should_run train; then
    set_dep "$LAST_DEP"
    PRIMARY_TRAIN="$(submit train-primary "${dep_args[@]}" --export="$EXPORTS,SCRIPTMIND_EXPERIMENT_SET=primary" \
        --job-name=scriptmind_train --array=0-5 "$SCRIPT_DIR/run_scriptmind_train.sbatch")"
    TRAIN_DEPS="$PRIMARY_TRAIN"
    if [[ "$WITH_ABLATIONS" == "1" ]]; then
        ABLATION_TRAIN="$(submit train-ablations "${dep_args[@]}" --export="$EXPORTS,SCRIPTMIND_EXPERIMENT_SET=ablations" \
            --job-name=scriptmind_ablation --array=0-8 "$SCRIPT_DIR/run_scriptmind_train.sbatch")"
        TRAIN_DEPS="${TRAIN_DEPS}:${ABLATION_TRAIN}"
    fi
    LAST_DEP="$TRAIN_DEPS"
fi

EVAL_DEPS=""
if should_run evaluate; then
    set_dep "$LAST_DEP"
    PRIMARY_EVAL="$(submit evaluate-primary "${dep_args[@]}" --export="$EXPORTS,SCRIPTMIND_EXPERIMENT_SET=primary" \
        --job-name=scriptmind_eval --array=0-15 "$SCRIPT_DIR/run_scriptmind_evaluate.sbatch")"
    EVAL_DEPS="$PRIMARY_EVAL"
    if [[ "$WITH_ABLATIONS" == "1" ]]; then
        ABLATION_EVAL="$(submit evaluate-ablations "${dep_args[@]}" --export="$EXPORTS,SCRIPTMIND_EXPERIMENT_SET=ablations" \
            --job-name=scriptmind_eval_ablation --array=0-17 "$SCRIPT_DIR/run_scriptmind_evaluate.sbatch")"
        EVAL_DEPS="${EVAL_DEPS}:${ABLATION_EVAL}"
    fi
    LAST_DEP="$EVAL_DEPS"
fi

if should_run report; then
    set_dep "$LAST_DEP"
    submit report "${dep_args[@]}" --export="$EXPORTS" --job-name=scriptmind_report \
        "$SCRIPT_DIR/run_scriptmind_cpu.sbatch" report >/dev/null
fi

echo "ScriptMind DAG processed (dry_run=$DRY_RUN, force=$FORCE, from=$FROM_STAGE, only=${ONLY_STAGE:-none}, ablations=$WITH_ABLATIONS)."
