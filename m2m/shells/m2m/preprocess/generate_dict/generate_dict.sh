LANGS=(af ar bg bn de el en es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl pt ru sw ta te th tl tr ur vi yo zh)
DICT=/path/to/NER/PretrainedModels/flores/flores101_mm100_615M/dict.txt
OUTPUT_DIR=/path/to/NER/flores/20M/data-bin-split20/

for ((i=0;i<=19;i++)); do
    for lg in ${LANGS[@]}; do
        echo "Copying $DICT -> $OUTPUT_DIR/data-bin${i}/dict.${lg}.txt"
        cp $DICT $OUTPUT_DIR/data-bin${i}/dict.${lg}.txt
    done
done