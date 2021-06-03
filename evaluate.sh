#!/bin/bash

#python utils/detokenizer.py ../data/$1/model_outputs/$2.0 ../data/$1/model_outputs/$2.0
#python utils/detokenizer.py ../data/$1/model_outputs/$2.1 ../data/$1/model_outputs/$2.1

echo "----------------BLEURT----------------"
python cal_bleurt.py ../data/$1/model_outputs/$2.0 ../data/$1/model_outputs/$2.1 \
                         ../data/$1/test/$3.ref ../data/$1/test/$4.ref

echo "----------------BLEU----------------"
python utils/tokenizer.py ../data/$1/model_outputs/$2.0 ../data/$1/outputs/$2.0 True
python utils/tokenizer.py ../data/$1/model_outputs/$2.1 ../data/$1/outputs/$2.1 True

perl utils/multi-bleu.perl ../data/$1/original_ref/$3.ref < ../data/$1/outputs/$2.0
perl utils/multi-bleu.perl ../data/$1/original_ref/$4.ref < ../data/$1/outputs/$2.1

echo "----------------Style----------------"
python classifier/test.py -dataset $1 -order $2
