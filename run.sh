#! /bin/bash

BASEDIR=../../lab_resources/DDI
export PYTHONPATH=$BASEDIR/util #Directory of the DDI data
export LD_LIBRARY_PATH=/home/martin/mambaforge/envs/mds/nvvm/libdevice:$LD_LIBRARY_PATH


if [[ "$*" == *"parse"* ]]; then
   $BASEDIR/util/corenlp-server.sh -quiet true -port 9000 -timeout 15000 &
   sleep 1

   python3 parse_data.py $BASEDIR/data/train train
   python3 parse_data.py $BASEDIR/data/devel devel
   python3 parse_data.py $BASEDIR/data/test  test
   kill `cat /tmp/corenlp-server.running`
fi

if [[ "$*" == *"train"* ]]; then
    rm -rf model*
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/martin/mambaforge/envs/mds python3 train.py train.pck devel.pck model
fi

if [[ "$*" == *"predict"* ]]; then
   rm -f devel.stats devel.out
   python3 predict.py model devel.pck devel.out 
   python3 $BASEDIR/util/evaluator.py DDI $BASEDIR/data/devel devel.out | tee devel.stats
fi

if [[ "$*" == *"test"* ]]; then
   rm -f test.stats test.out
   python3 predict.py model test.pck test.out 
   python3 $BASEDIR/util/evaluator.py DDI $BASEDIR/data/test test.out | tee test.stats
fi


