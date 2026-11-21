#!/bin/bash
cd /mnt/data/inklsa/runs
for a in "16384 256" "32768 256" "32768 1024" "131072 256" "131072 64" "262144 64" "262144 16"; do ./cbq.sh $a; done
