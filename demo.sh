#!/usr/bin/env bash

# This is a demo to run CNN-BiLSTM-CRF with Prism Module in CoNLL-2003 NER task.

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove.6B
python3 data/process.py
CUDA_VISIBLE_DEVICES=0 python3 run.py

