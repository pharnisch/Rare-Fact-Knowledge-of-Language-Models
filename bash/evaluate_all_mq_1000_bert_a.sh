#!/bin/bash

for RELATION in P127 P131 P136 P138 P140 P159 P176 P178 P190
do
  python eval.py bert-base-cased_pretrained $RELATION -mq=1000
done