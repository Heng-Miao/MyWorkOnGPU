#!/bin/bash
#PBS -N ampt
#PBS -l nodes=1:ppn=1
#PBS -l walltime=18:00:00
#PBS -e log/job1.err
#PBS -o log/job1.out
#PBS -M lgpang.1984@gmail.com
#PBS -m e
#PBS -m a
#PBS -r n
cd /u/phys/lgpang/Ini
echo "Job start at time"
echo `date`
sh ./exec 
echo "Job end at time"
echo `date`
