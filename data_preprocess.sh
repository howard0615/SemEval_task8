#!/bin/bash

python preprocess.py --data_dir ./data --train True --test True 

python split_to_5_fold.py --data_dir ./data
