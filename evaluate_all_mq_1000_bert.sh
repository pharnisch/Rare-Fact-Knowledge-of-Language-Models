#!/bin/bash

for RELATION in test date_of_birth place_of_birth place_of_death P17 P19 P20 P27 P30 P31 P36 P37 P39 P47 P101 P103 P106 P108
do
  python eval.py bert-base-cased_pretrained $RELATION -mq=1000
done