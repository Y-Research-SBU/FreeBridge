#!/usr/bin/env bash
set -euo pipefail

command -v wget >/dev/null || { echo "ERROR: wget not found"; exit 1; }
command -v hf >/dev/null   || { echo "ERROR: hf CLI not found (pip install huggingface_hub)"; exit 1; }

W_RAW="${1:?Usage: $0 <week 1..10>}"
W=$((10#$W_RAW))   # force decimal so 08/09 don't become invalid octal
if (( W < 1 || W > 10 )); then
  echo "ERROR: week must be in 1..10, got ${W_RAW}"; exit 1
fi
W_PAD="$(printf '%02d' "$W")"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPTS="${SCRIPT_DIR}"
RAW_LIST="${SCRIPT_DIR}/raw_list.txt"
[[ -f "${RAW_LIST}" ]] || { echo "ERROR: missing ${RAW_LIST}"; exit 1; }
CROP_SCRIPT="${SCRIPT_DIR}/crop_bbbc021_singlecell.py"

OUT_ROOT="${ROOT}/data/crops/week${W_PAD}"
TMP_DIR="${ROOT}/tmp"
RAW_DIR="${TMP_DIR}/raw_week${W}"
MASK_DIR="${TMP_DIR}/mask_week${W}"
HF_DIR="${ROOT}/hf"

BROAD_BASE="https://data.broadinstitute.org/bbbc/BBBC021"
MASK_TAR="mask_week${W}.tar.gz"

mkdir -p "${OUT_ROOT}" "${TMP_DIR}" "${HF_DIR}"

echo "=============================="
echo "WEEK ${W}"
echo "=============================="

echo "[RAW] collect zip list from raw_list.txt..."
mapfile -t ZIPS < <(grep -E "^BBBC021_v1_images_Week${W}_.+\.zip$" "${RAW_LIST}" || true)
if [[ "${#ZIPS[@]}" -eq 0 ]]; then
  echo "ERROR: no raw zips found for Week ${W} in ${RAW_LIST}"
  exit 2
fi
printf "  %s\n" "${ZIPS[@]}"

echo "[RAW] download + unzip..."
rm -rf "${RAW_DIR}"
mkdir -p "${RAW_DIR}/unzipped"
cd "${RAW_DIR}"

for z in "${ZIPS[@]}"; do
  echo "  wget $z"
  wget -q --show-progress -c "${BROAD_BASE}/${z}"
  unzip -q "${z}" -d "${RAW_DIR}/unzipped"
  rm -f "${z}"
done

RAW_ROOT="${RAW_DIR}/unzipped"
echo "RAW_ROOT=${RAW_ROOT}"

echo "[MASK] download from HF..."
cd "${HF_DIR}"

# Download masks (avoid --local-dir-use-symlinks for hf CLI compatibility)
hf download "CurioWang/BBBC021" \
  --repo-type dataset \
  --local-dir "${HF_DIR}" \
  --include "${MASK_TAR}"

if [[ ! -f "${HF_DIR}/${MASK_TAR}" ]]; then
  echo "ERROR: mask tar not found after hf download: ${HF_DIR}/${MASK_TAR}"
  ls -lh "${HF_DIR}" | tail -n 50 || true
  exit 4
fi

echo "[MASK] extract..."
rm -rf "${MASK_DIR}"
mkdir -p "${MASK_DIR}"
tar -xzf "${HF_DIR}/${MASK_TAR}" -C "${MASK_DIR}"
find "${MASK_DIR}" -name '._*' -delete || true

# Locate the directory that actually contains the mask TIFFs
MASK_ROOT="${MASK_DIR}"
if ! find "${MASK_ROOT}" -maxdepth 1 -type f -name "*mask*.tif*" | grep -q .; then
  CAND="$(find "${MASK_DIR}" -type f -name "*mask*.tif*" -print0 | xargs -0 -n1 dirname | sort -u | head -n 1 || true)"
  if [[ -n "${CAND}" ]]; then
    MASK_ROOT="${CAND}"
  fi
fi
echo "MASK_ROOT=${MASK_ROOT}"

echo "[CROP] run crop..."
python "${CROP_SCRIPT}" \
  --raw_root "${RAW_ROOT}" \
  --mask_root "${MASK_ROOT}" \
  --out_root "${OUT_ROOT}"

echo "[CLEANUP] remove week raw/mask/tar..."
rm -rf "${RAW_DIR}"
rm -rf "${MASK_DIR}"
rm -f "${HF_DIR}/${MASK_TAR}"

echo "[DONE] Week ${W} -> ${OUT_ROOT}"
