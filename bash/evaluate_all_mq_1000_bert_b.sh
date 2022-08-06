#!/bin/bash

for RELATION in P264 P276 P279 P361 P364 P407 P413 P449 P463
do
  python eval.py bert-base-cased_pretrained $RELATION -mq=1000
done