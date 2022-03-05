
# [Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer (ACL2021)](https://arxiv.org/abs/2105.06947)

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
python classifier/textcnn.py -dataset em
```
**Note:** the style classifiers are already provided in checkpoints.

### Step 2: Fine-tuning BART
```
sh pg.sh em 0.0 sc bl formal informal
```
**Note:** in the paper, we employ the style classifier that uses GPT2 tokenizer to evaluate style strength.

## System Output
The outputs of our best systems (BART (large + rewards) trained with the combined domains data are provided in outputs.

## Cite
If you use this code, please cite our paper:
```
@inproceedings{lai-etal-2021-thank,
    title = "Thank you {BART}! Rewarding Pre-Trained Models Improves Formality Style Transfer",
    author = "Lai, Huiyuan and Toral, Antonio and Nissim, Malvina",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.62",
    doi = "10.18653/v1/2021.acl-short.62",
    pages = "484--494",
}
```
