export PYTHONWARNINGS="ignore"

dir=$1
MODEL=$2
batchsize=$3
beam=$4
TEST_SHELL=./shells/aml/multi-node/large_task/deltalm/test/test_aml_devtest.sh

echo $dir

    src=en
    TGTS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl no pt ru sw ta te th tl tr ur vi yo zh)
    for tgt in "${TGTS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
    done

    SRCS=(af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my nl no pt ru sw ta te th tl tr ur vi yo zh)
    tgt=en
    for src in "${SRCS[@]}"; do
        echo "${src}->${tgt}"
        bash $TEST_SHELL $src $tgt $batchsize $beam $MODEL
    done
