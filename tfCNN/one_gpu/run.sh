# USAGE: 
#  1. Copy in dir with network script .py
#  2. Run from lmbtorque using command: bash run.sh

dirname=`dirname $(readlink -f $0)`
pyscript=cnn_train.py

sed -e "s+@dirname@+${dirname}+g" \
    -e "s+@pyscript@+${pyscript}+g" > run-train.sh <<EOF
#!/bin/bash
#PBS -N net1-train
#PBS -S /bin/bash
#PBS -l nodes=1:nvidiaTITANX:ppn=1,gpus=1,mem=11000mb,walltime=23:00:00
#PBS -M d.d.tananaev@gmail.com
#PBS -m bea
#PBS -j oe
#PBS -o out.txt
#PBS -V
cd @dirname@
source /misc/lmbraid11/tananaed/tf/bin/activate
python @pyscript@ 2>&1| tee out.log
EOF
qsub run-train.sh
