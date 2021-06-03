# -*- coding: utf-8 -*-

import sys
import string

punc = string.punctuation
punc += "n't，《。》、？；：‘”【「】」、|·~！@#￥%……&*（）-——+='s'm'll''``"
import sys
import nltk
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer

punc = string.punctuation

seqs = []
with open(sys.argv[1],'r') as f:
    for line in f.readlines():
        seqs.append(line.strip())

with open(sys.argv[2],'w') as f:
    for line in seqs:
        line = nltk.word_tokenize(line.strip())
        line = TreebankWordDetokenizer().detokenize(line)
        tokens = line.split()
        line = ''
        for token in tokens:
            if token not in punc:
                line += (' '+token)
            else:
                line += token
#        line = line.strip().capitalize()
        f.write(line.strip()+'\n')

