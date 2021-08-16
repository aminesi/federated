#! /bin/bash

for CONF_ID in 26 32 33 37 40 77 109 124 125 161 163 173 179 182 187 194 196 201 203 214 221 231 234 242 243 244 247 ; do
   sbatch --export=ALL,CONF_ID=$CONF_ID --job-name=fed-job-$CONF_ID cc/run.sh
done
