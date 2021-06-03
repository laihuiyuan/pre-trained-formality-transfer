# -*- coding: utf-8 -*-

import random
random.seed(1024)
import numpy as np

import torch
import torch.utils.data


def read_data(dataset, style, max_len, prefix,
              tokenizer, domain=0, ratio=1.0):

    if domain!=0:
        domain = tokenizer.encode(domain, add_special_tokens=False)[0]

    if style == 0:
        src_file = '../data/{}/{}.0'.format(dataset, prefix)
        tgt_file = '../data/{}/{}.1'.format(dataset, prefix)
    else:
        src_file = '../data/{}/{}.1'.format(dataset, prefix)
        tgt_file = '../data/{}/{}.0'.format(dataset, prefix)

    src_seq, tgt_seq = [], []
    with open(src_file, 'r') as f1, open(tgt_file, 'r') as f2:
        f1 = f1.readlines()
        f2 = f2.readlines()
        index = [i for i in range(len(f1))]
        random.shuffle(index)
        index = index[:int(len(index) * ratio)]
        for i, (s, t) in enumerate(zip(f1, f2)):
            if i in index:
                s = tokenizer.encode(s)
                t = tokenizer.encode(t)
                s = s[:min(len(s) - 1, max_len)] + s[-1:]
                t = t[:min(len(t) - 1, max_len)] + t[-1:]
                s[0] = domain
                src_seq.append(s)
                tgt_seq.append([tokenizer.bos_token_id]+t)

    return src_seq, tgt_seq


def collate_fn(insts, pad_token_id=1):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)
    max_len = max_len if max_len > 4 else 5

    batch_seq = np.array([
        inst + [pad_token_id] * (max_len - len(inst))
        for inst in insts])
    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


def paired_collate_fn(insts):
    src_inst, tgt_inst = list(zip(*insts))
    src_inst = collate_fn(src_inst)
    tgt_inst = collate_fn(tgt_inst)

    return src_inst, tgt_inst


class CNNDataset(torch.utils.data.Dataset):
    def __init__(self, insts, label):
        self.insts = insts
        self.label = label

    def __getitem__(self, index):
        return self.insts[index], self.label[index]

    def __len__(self):
        return len(self.insts)


def SCIterator(insts_0, insts_1, opt, pad_token_id=1, shuffle=True):
    '''Data iterator for style classifier'''

    def cls_fn(insts):
        insts, labels = list(zip(*insts))
        seq = collate_fn(insts, pad_token_id)
        labels = torch.LongTensor(labels)
        return (seq, labels)

    num = len(insts_0) + len(insts_1)
    loader = torch.utils.data.DataLoader(
        CNNDataset(
            insts=insts_0 + insts_1,
            label=[0 if i < len(insts_0)
                   else 1 for i in range(num)]),
        shuffle=shuffle,
        num_workers=2,
        collate_fn=cls_fn,
        batch_size=opt.batch_size)

    return loader


def load_embedding(tokenizer, embed_dim, embed_path=None):
    '''Parse an embedding text file into an array.'''

    embedding = np.random.normal(scale=embed_dim ** -0.5,
                                 size=(len(tokenizer), embed_dim))
    if embed_path == None:
        return embedding

    print('[Info] Loading embedding')
    embed_dict = {}
    with open(embed_path) as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            tokens = line.rstrip().split()
            try:
                embed_dict[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                continue

    for i in range(len(tokenizer)):
        try:
            word = tokenizer.decode(i)
            if word in embed_dict:
                embedding[i] = embed_dict[word]
        except:
            print(i)

    return embedding


class BartDataset(torch.utils.data.Dataset):
    def __init__(self, src_inst=None, tgt_inst=None):
        self._src_inst = src_inst
        self._tgt_inst = tgt_inst

    def __len__(self):
        return len(self._src_inst)

    def __getitem__(self, idx):
        return self._src_inst[idx], self._tgt_inst[idx]


def BARTIterator(train_src, train_tgt,
                 valid_src, valid_tgt, opt):
    '''Data iterator for fine-tuning BART'''

    train_loader = torch.utils.data.DataLoader(
        BartDataset(
            src_inst=train_src,
            tgt_inst=train_tgt),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        BartDataset(
            src_inst=valid_src,
            tgt_inst=valid_tgt),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader