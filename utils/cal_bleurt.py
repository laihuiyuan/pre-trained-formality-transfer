# -*- coding: utf-8 -*-

import sys
from bleurt import score


checkpoint = '../checkpoints/bleurt-base-128'
scorer = score.BleurtScorer(checkpoint)

def cal_bleurt(file0, file1):
    scores = []
    with open(file0,'r') as fin:
        hyps = []
        for line in fin.readlines():
            hyps.append(line.strip())
    for i in range(4):
        with open(file1+str(i),'r') as fin:
            refs = []
            for line in fin.readlines():
                refs.append(line.strip())
            scores.extend(scorer.score(refs, hyps))
    return scores

scores = []
scores.extend(cal_bleurt(sys.argv[1],sys.argv[3]))
scores.extend(cal_bleurt(sys.argv[2],sys.argv[4]))
print('The average bleurt score is {}'.format(sum(scores)/len(scores)))
