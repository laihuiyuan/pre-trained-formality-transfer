
# [Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer (ACL-IJCNLP 2021)](https://arxiv.org/abs/2105.06947)

## Dependencies
```
python==3.7
pytorch==1.4.0
transformers==2.5.1
```

## Dataset
### [GYAFC](https://github.com/raosudha89/GYAFC-corpus): informal text (0) <-> formal text (1)
The GYAFC dataset is only free of charge for research purposes, you can follow the guidance [https://github.com/raosudha89/GYAFC-corpus](https://github.com/raosudha89/GYAFC-corpus) to get the dataset.

## Quick Start
### Step 1: Pre-train style classifier
```
cd classifier
python textcnn.py -dataset em
```
**Note:** the style classifiers are already provided in checkpoints.

### Step 2: Fine-tuning BART
```
cd classifier
sh pg_traing.sh em 0.0 sc bl formal informal
```
**Note:** in the paper, we employ the style classifier that uses GPT2 tokenizer to evaluate style strength.

## System Output
The outputs of our best systems (BART (large + rewards) trained with the combined domains data are provided in outputs.

## Cite
If you use this code, please cite the following paper:
```
@article{lai-etal-2021,
    title={{Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer}}, 
      author={Huiyuan Lai and Antonio Toral and Malvina Nissim},
      year={2021},
      archivePrefix={arXiv},
      eprint={2105.06947},
      primaryClass={cs.CL},
      journal={arXiv preprint, arXiv: 2105.06947}
}