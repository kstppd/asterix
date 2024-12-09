#!/bin/bash
# Modify target JOBID below, then execute to run the inspector.sh on every node for every srun vlasiator instance.

export JOBID=8513363

squeue --job $JOBID -o "%N" | grep nid | cut -b 5- | cut --delimiter "]" -f 1 > .nodelist

python3 - $( cat .nodelist ) > .fullnodelist << EOS
import sys

nodes_str_list = sys.argv[1].split(",")
outnodes = []

for n in nodes_str_list:
    if "-" in n:
        n_range = n.split("-")
        for i in range(int(n_range[0]),int(n_range[1])+1):
            outnodes.append("nid"+str(i).zfill(6))
    else:
        outnodes.append("nid"+n)

for t in outnodes:
    print(str(t))

EOS


for node in $( cat .fullnodelist )
do
  export NODE=$node
  srun --jobid $JOBID --overlap -w $node bash ./inspector.sh &
done

wait
