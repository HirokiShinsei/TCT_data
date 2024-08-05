#!/bin/bash
# run froc.py
TYPE="1"
GT_CSV='/home/stat-zx/TCT_FIFTH/test.csv'


python froc.py \
    --type $TYPE \
    $GT_CSV
    
