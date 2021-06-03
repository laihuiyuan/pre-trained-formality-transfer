# -*- coding: utf-8 -*-

import sys
import nltk

fin = open(sys.argv[1],'r').readlines()
with open(sys.argv[2],'w') as f:
    for line in fin:
        if sys.argv[3]=='True':
            line = nltk.word_tokenize(line.strip().lower())
        else:
            line = nltk.word_tokenize(line.strip())
        f.write(' '.join(line)+'\n')
