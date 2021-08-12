#! /bin/bash

for CONF_ID in 16 26 27 28 29 30 31 32 33 34 36 37 38 39 40 41 42 45 47 48 49 63 73 ; do
   sbatch --export=ALL,CONF_ID=$CONF_ID --job-name=fed-job-$CONF_ID cc/run.sh
done

for CONF_ID in {75..300} ; do
   sbatch --export=ALL,CONF_ID=$CONF_ID --job-name=fed-job-$CONF_ID cc/run.sh
done