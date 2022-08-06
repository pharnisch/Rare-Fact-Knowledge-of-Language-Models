#!/bin/bash

for MODEL in bert-base-cased_pretrained CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth CorDistilBert-12-1.0-4096-0.000100-9-1.693901-0.6584-checkpoint.pth
do
  python eval.py $MODEL P1303 -mq=1000
done