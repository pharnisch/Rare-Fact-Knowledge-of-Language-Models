#!/bin/bash

for RELATION in test date_of_birth place_of_birth place_of_death P17 P19 P20 P27 P30 P31 P36 P37 P39 P47 P101 P103 P106 P108 P127 P131 P136 P138 P140 P159 P176 P178 P190 P264 P276 P279 P361 P364 P407 P413 P449 P463 P495 P527 P530 P740 P937 P1001 P1303 P1376 P1412
do
  python eval.py CorBert-12-1-4096-0.000500-9-1.359077-0.713-checkpoint.pth $RELATION -mq=1000
done