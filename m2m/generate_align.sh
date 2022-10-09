ORIG_TRAIN="/path/to/data/xlmr-data/spm-data/"
TRAIN="/path/to/data/xlmr-data/spm-data-filter/"
NEW_TRAIN="/path/to/alignment/train/"

ALIGN="/path/to/fast_align/build/fast_align"
ATOOLS="/path/to/fast_align/build/atools"
FILTER="/path/to/xlm-t/filter.py"
# PREPROCESS="/path/to/xlm-t/fairseq_cli/preprocess.py"

EM_STEP=10

mkdir -p "${ORIG_TRAIN} ${TRAIN}"
for lp in "en-fr" "en-de" "en-fi" "en-cs" "en-et" "en-tr" "en-lv" "en-ro" "en-hi" "en-gu"; do
  src=$(echo ${lp} | cut -d "-" -f 1)
  tgt=$(echo ${lp} | cut -d "-" -f 2)
  for splt in valid train; do
    echo "Start aligning ${splt}: ${src}-${tgt} "
    python $FILTER -src-set "${ORIG_TRAIN}/$splt.${src}-${tgt}.${src}" \
      -tgt-set "${ORIG_TRAIN}/${splt}.${src}-${tgt}.${tgt}" \
      -new-src-set "${TRAIN}/$splt.${src}-${tgt}.${src}" \
      -new-tgt-set "${TRAIN}/$splt.${src}-${tgt}.${tgt}"
    paste "${TRAIN}/$splt.${src}-${tgt}.${src}" "${TRAIN}/$splt.${src}-${tgt}.${tgt}" |
      awk -F '\t' '{print $1 " ||| " $2}' >"${NEW_TRAIN}/$splt.${src}-${tgt}"
    $ALIGN -i "${NEW_TRAIN}/$splt.${src}-${tgt}" -d -o -v -I ${EM_STEP} \
      >"${NEW_TRAIN}/$splt.${src}-${tgt}.forward.align"
    $ALIGN -i "${NEW_TRAIN}/$splt.${src}-${tgt}" -d -o -v -r -I ${EM_STEP} \
      >"${NEW_TRAIN}/$splt.${src}-${tgt}.backward.align"
    $ATOOLS -i "${NEW_TRAIN}/$splt.${src}-${tgt}.forward.align" -j \
      "${NEW_TRAIN}/$splt.${src}-${tgt}.backward.align" -c grow-diag-final-and \
      >"${NEW_TRAIN}/$splt.${src}-${tgt}.align"
    rm "${NEW_TRAIN}/$splt.${src}-${tgt}.forward.align" "${NEW_TRAIN}/$splt.${src}-${tgt}.backward.align" \
      "${NEW_TRAIN}/$splt.${src}-${tgt}"
  done
done
