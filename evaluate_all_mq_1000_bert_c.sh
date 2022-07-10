#!/bin/bash

for RELATION in P495 P527 P530 P740 P937 P1001 P1301 P1376 P1412
do
  python eval.py bert-base-cased_pretrained $RELATION -mq=1000
done