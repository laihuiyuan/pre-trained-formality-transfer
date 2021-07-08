#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=50000

python train.py -style 0 -ratio 1.0 -dataset $1 -order $2.0 -$3 -$4
python infer.py -style 0 -dataset $1 -order $2.0
rm checkpoints/bart_$1_$2.0_0.chkpt

python train.py -style 1 -ratio 1.0 -dataset $1 -order $2.0 -$3 -$4
python infer.py -style 1 -dataset $1 -order $2.0
rm checkpoints/bart_$1_$2.0_1.chkpt

echo "----------------Style----------------"
python classifier/test.py -dataset $1 -order $2.0

echo "----------------BLEU----------------"
python utils/tokenizer.py data/outputs/bart_$1_$2.0.0 data/$1/outputs/bart_$1_$2.0.0 False
python utils/tokenizer.py data/outputs/bart_$1_$2.0.1 data/$1/outputs/bart_$1_$2.0.1 Flase
perl utils/multi-bleu.perl data/$1/original_ref/$5.ref < data/$1/outputs/bart_$1_$2.0.0
perl utils/multi-bleu.perl data/$1/original_ref/$6.ref < data/$1/outputs/bart_$1_$2.0.1

echo "----------------BLEURT----------------"
python utils/cal_bleurt.py data/outputs/bart_$1_$2.0.0 data/outputs/bart_$1_$2.0.1 \
                         data/$1/test/$5.ref data/$1/test/$6.ref

