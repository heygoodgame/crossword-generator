#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./generate-puzzles.sh WEEKS [options] [-- extra generate-pilot-batch args]

Generates live-slot crossword batches at the Mini/Midi weekly ratio:
  per week: 5 Mini 5x5, 2 Mini 7x7, 7 Midi 9x9

Options:
  --batch-id ID          Batch id to record in the manifest.
  --output-root DIR      Output root. Defaults to output/batches/<batch-id>.
  --seed SEED            First deterministic seed. Defaults to a random seed.
  --difficulty LEVEL     easy or hard. Defaults to easy.
  --all-difficulties     Generate both easy and hard buckets.
  --buckets LIST         Explicit bucket list, e.g. easy/5,easy/7,easy/9.
  --llm PROVIDER         claude or ollama. Defaults to claude.
  --avoid-existing-clues Load existing generated clues before generation.
  --dry-run              Print the resolved command without running it.
  -h, --help             Show this help.
USAGE
}

random_seed() {
  local raw
  if command -v od >/dev/null 2>&1; then
    raw="$(od -An -N4 -tu4 /dev/urandom | tr -d '[:space:]')"
  else
    raw="$(date -u +%s)"
  fi
  echo $((raw % 900000 + 100000))
}

shell_quote() {
  printf '%q' "$1"
}

weeks="${1:-}"
if [[ -z "${weeks}" || "${weeks}" == "-h" || "${weeks}" == "--help" ]]; then
  usage
  exit 0
fi
shift

if ! [[ "${weeks}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WEEKS must be a positive integer; got '${weeks}'." >&2
  exit 2
fi

difficulty="easy"
all_difficulties=false
buckets=""
batch_id=""
output_root=""
seed_start=""
llm="claude"
avoid_existing_clues=false
dry_run=false
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-id)
      batch_id="${2:-}"
      shift 2
      ;;
    --output-root)
      output_root="${2:-}"
      shift 2
      ;;
    --seed|--seed-start)
      seed_start="${2:-}"
      shift 2
      ;;
    --difficulty)
      difficulty="${2:-}"
      shift 2
      ;;
    --all-difficulties)
      all_difficulties=true
      shift
      ;;
    --buckets)
      buckets="${2:-}"
      shift 2
      ;;
    --llm)
      llm="${2:-}"
      shift 2
      ;;
    --avoid-existing-clues)
      avoid_existing_clues=true
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${difficulty}" != "easy" && "${difficulty}" != "hard" ]]; then
  echo "--difficulty must be 'easy' or 'hard'; got '${difficulty}'." >&2
  exit 2
fi
if [[ "${llm}" != "claude" && "${llm}" != "ollama" ]]; then
  echo "--llm must be 'claude' or 'ollama'; got '${llm}'." >&2
  exit 2
fi
if [[ -n "${seed_start}" && ! "${seed_start}" =~ ^[0-9]+$ ]]; then
  echo "--seed must be a non-negative integer; got '${seed_start}'." >&2
  exit 2
fi

count_5=$((weeks * 5))
count_7=$((weeks * 2))
count_9=$((weeks * 7))
bucket_counts="5=${count_5},7=${count_7},9=${count_9}"

if [[ -z "${seed_start}" ]]; then
  seed_start="$(random_seed)"
fi
if [[ -z "${batch_id}" ]]; then
  batch_id="weekly-${weeks}w-$(date -u +%Y%m%d-%H%M%S)-s${seed_start}"
fi
if [[ -z "${output_root}" ]]; then
  output_root="output/batches/${batch_id}"
fi
if [[ -z "${buckets}" ]]; then
  if [[ "${all_difficulties}" == true ]]; then
    buckets="easy/5,easy/7,easy/9,hard/5,hard/7,hard/9"
  else
    buckets="${difficulty}/5,${difficulty}/7,${difficulty}/9"
  fi
fi

cmd=(
  uv run crossword-generator generate-pilot-batch
  --output-root "${output_root}"
  --batch-id "${batch_id}"
  --buckets "${buckets}"
  --bucket-counts "${bucket_counts}"
  --seed-start "${seed_start}"
  --llm "${llm}"
)
if [[ "${avoid_existing_clues}" == true ]]; then
  cmd+=(--avoid-existing-clues)
fi
if [[ ${#extra_args[@]} -gt 0 ]]; then
  cmd+=("${extra_args[@]}")
fi

echo "Weeks: ${weeks}"
echo "Bucket counts: ${bucket_counts}"
echo "Buckets: ${buckets}"
echo "Batch id: ${batch_id}"
echo "Output root: ${output_root}"
echo "Seed start: ${seed_start}"
echo "LLM: ${llm}"
printf 'Command:'
for part in "${cmd[@]}"; do
  printf ' %s' "$(shell_quote "${part}")"
done
printf '\n'

if [[ "${dry_run}" == true ]]; then
  exit 0
fi

"${cmd[@]}"
