#!/usr/bin/env bash
set -e

# -----------------------------------------------------------------------------
# EarthScape Dataset Release Script (for MacOS)
#
# Creates the following:
#   - One ZIP per immediate subdirectory in earthscape/data/ (Data subsets)
#   - One ZIP for top-level files in earthscape/data/ (Full dataset metadata)
#   - One ZIP for docs/metadata at repo root (README, CHANGELOG)
#   - One SHA256SUMS.txt containing SHA-256 checksums for all zip files.
#  
# -----------------------------------------------------------------------------

# resolve paths relative to this script...
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
OUT_DIR="$ROOT_DIR/release"


# create output directory and remove any previous artifacts.
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/*.zip "$OUT_DIR"/SHA256SUMS.txt


# -----------------------------------------------------------------------------
# 1) Zip each immediate subdirectory under earthscape/data/
#    We cd into DATA_DIR so each zip has clean paths like:
#      subset_name/... (not /full/path/to/subset_name/...)
# -----------------------------------------------------------------------------
for d in "$DATA_DIR"/*/; do
  [ -d "$d" ] || continue
  subset="$(basename "$d")"
  zip_path="$OUT_DIR/earthscape_data_${subset}.zip"

  ( cd "$DATA_DIR" && zip -r -X -q "$zip_path" "$subset" )
done

# -----------------------------------------------------------------------------
# 2) Zip ONLY the top-level files in earthscape/data/ (exclude directories)
#    We generate a list of just those files and pass it to zip via stdin (-@).
# -----------------------------------------------------------------------------
global_zip="$OUT_DIR/earthscape_data_global_files.zip"
(
  cd "$DATA_DIR"
  find . -mindepth 1 -maxdepth 1 -type f -print \
    | sed 's|^\./||' \
    | zip -X -q "$global_zip" -@
)

# -----------------------------------------------------------------------------
# 3) Zip documentation/metadata files (repo root)
#    Add/remove files here as needed.
# -----------------------------------------------------------------------------
docs_zip="$OUT_DIR/earthscape_docs_metadata.zip"
DOC_FILES=(README.md CHANGELOG.md CITATION.cff)

# Only include doc files that exist (CITATION.cff may be optional).
DOC_EXISTING=()
for f in "${DOC_FILES[@]}"; do
  [ -f "$ROOT_DIR/$f" ] && DOC_EXISTING+=("$f")
done

( cd "$ROOT_DIR" && zip -r -X -q "$docs_zip" "${DOC_EXISTING[@]}" )

# -----------------------------------------------------------------------------
# 4) SHA-256 checksums for all zip files
#    Format: "<hash>  <filename>"
# -----------------------------------------------------------------------------
sums_file="$OUT_DIR/SHA256SUMS.txt"
: > "$sums_file"

for z in "$OUT_DIR"/*.zip; do
  printf "%s  %s\n" "$(shasum -a 256 "$z" | awk '{print $1}')" "$(basename "$z")" >> "$sums_file"
done

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
echo "Release artifacts written to: $OUT_DIR"
ls -1 "$OUT_DIR"


# zip -r -X esv1p1_smokeset.zip esv1p1_smoke -x "*.DS_Store"
