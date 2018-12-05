#!/bin/bash
#PBS -N ampt_ridge
#PBS -q lr_batch
#PBS -A ac_jet
#PBS -l nodes=1:ppn=1
#PBS -l walltime=00:20:00
#PBS -e log/job2.err
#PBS -o log/job2.out
#PBS -m e
#PBS -m a
#PBS -r n
cd /global/home/users/lpang/ampt-v1.21-v2.21
###mkdir ana/ampt_b6-15_Ini/
g++ ridge.cpp -O3 -o ridge
echo "Job start at time"
echo `date`
./ridge 
echo "Job end at time"
echo `date`
