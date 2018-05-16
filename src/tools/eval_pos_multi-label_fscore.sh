#!/bin/bash
python `dirname "$0"`/eval_pos_fscore.py -gold $1 -auto $2  
python `dirname "$0"`/eval_pos_fscore.py -gold $1 -auto $2 -xpos
