#!/bin/bash
for a in 10 20 50 100 200 300 350 360 365 366 367 368 369
do
    python process/visualize.py -i $a -c $1 # $1 is command parameter.
done
