# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
from torch import cuda
from torch.nn import CrossEntropyLoss

from model import BartModel
from model import BartForMaskedLM
from transformers import BartTokenizer
from transformers.modeling_bart import make_padding_mask

from classifier.textcnn import TextCNN
from utils.optim import ScheduledOptim
from utils.helper import optimize, evaluate
from utils.helper import cal_sc_loss, cal_bl_loss
from utils.dataset import read_data, BARTIterator

device = 'cuda' if cuda.is_available() else 'cpu'

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]


def main():
    parser = argparse.ArgumentParser('Fine-Tuned BART for style transfer')
    parser.add_argument('-no', action='store_true', help='no reward')
    parser.add_argument('-sc', action='store_true', help='the SC-based reward')
    parser.add_argument('-bl', action='store_true', help='the BLEU-based reward')
    parser.add_argument('-order', default=0, type=str, help='the order of traing')
    parser.add_argument('-style', default=0, type=int, help='transfer inf. to for.')
    parser.add_argument('-lr', default=1e-5, type=float, help='the learning rate')
    parser.add_argument('-ratio', default=1., type=float, help='proportion of data')
    parser.add_argument('-model', default='bart', type=str, help='the name of model')
    parser.add_argument('-dataset', default='em', type=str, help='the name of dataset')
    parser.add_argument('-steps', default=20000, type=int, help='force stop at x steps')
    parser.add_argument('-batch_size', default=32, type=int, help='the size in a batch')
    parser.add_argument('-max_len', default=30, type=int, help='maximum tokens a batch')
    parser.add_argument('-dropout', default=0.5, type=float, help='Keep prob in dropout')
    parser.add_argument('-patience', default=2, type=int, help='early stopping fine-tune')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random generator seed')
    parser.add_argument('-log_step', default=100, type=int, help='print logs every x step')
    parser.add_argument('-eval_step', default=1000, type=int, help='evaluate every x step')

    opt = parser.parse_args()
    print('[Info]', opt)
    torch.manual_seed(opt.seed)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    # for token in ['<E>', '<F>']:
    #     tokenizer.add_tokens(token)

    base = BartModel.from_pretrained("facebook/bart-base")
    model = BartForMaskedLM.from_pretrained('facebook/bart-base', config=base.config)
    # model.resize_token_embeddings(len(tokenizer))
    # model.base_model.encoder.embed_tokens=model.base_model.shared
    # model.base_model.decoder.embed_tokens=model.base_model.shared
    # model.lm_head=_make_linear_from_emb(model.base_model.shared)
    model.to(device).train()

    cls = TextCNN(300, len(tokenizer), filter_sizes,
                  num_filters, None, dropout=opt.dropout)
    cls.to(device).eval()
    cls.load_state_dict(torch.load('checkpoints/textcnn_{}.chkpt'.format(
                        opt.dataset)))

    train_src, train_tgt, = read_data(opt.dataset, opt.style, opt.max_len,
                                      'train', tokenizer, ratio=opt.ratio)
    valid_src, valid_tgt = read_data(opt.dataset, opt.style, opt.max_len,
                                     'valid', tokenizer)
    print('[Info] {} instances from train set'.format(len(train_src)))
    print('[Info] {} instances from valid set'.format(len(valid_tgt)))

    train_loader, valid_loader = BARTIterator(train_src, train_tgt,
                                              valid_src, valid_tgt, opt)

    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                         betas=(0.9, 0.98), eps=1e-09), opt.lr, 10000)

    tab = 0
    eval_loss = 1e8
    total_loss_ce = []
    total_loss_sc = []
    total_loss_co = []
    start = time.time()
    train_iter = iter(train_loader)
    for step in range(1, opt.steps):

        try:
            batch = next(train_iter)
        except:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        src, tgt = map(lambda x: x.to(device), batch)
        src_mask = make_padding_mask(src, tokenizer.pad_token_id)
        src_mask = 1 - src_mask.long() if src_mask is not None else None
        logits = model(src, attention_mask=src_mask, decoder_input_ids=tgt)[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgt[..., 1:].contiguous()
        loss_ce = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                          shift_labels.view(-1))
        total_loss_ce.append(loss_ce.item())

        loss_sc, loss_co = torch.tensor(0), torch.tensor(0)
        if opt.sc and (200 < step or len(train_loader) < step):
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
            loss_sc = cal_sc_loss(logits, idx, cls, tokenizer, opt.style)
            total_loss_sc.append(loss_sc.item())
        if opt.bl and (200 < step or len(train_loader)< step):
            idx = tgt.ne(tokenizer.pad_token_id).sum(-1)
            loss_co = cal_bl_loss(logits, tgt, idx, tokenizer)
            total_loss_co.append(loss_co.item())

        optimize(optimizer, loss_ce + loss_sc + loss_co)

        if step % opt.log_step == 0:
            lr = optimizer._optimizer.param_groups[0]['lr']
            print('[Info] steps {:05d} | loss_ce {:.4f} | loss_sc {:.4f} | '
                  'loss_co {:.4f} | lr {:.6f} | second {:.2f}'.format(
                step, np.mean(total_loss_ce), np.mean(total_loss_sc),
                np.mean(total_loss_co), lr, time.time() - start))
            total_loss_ce = []
            total_loss_sc = []
            total_loss_co = []
            start = time.time()

        if ((len(train_loader) > opt.eval_step
             and step % opt.eval_step == 0)
                or (len(train_loader) < opt.eval_step
                    and step % len(train_loader) == 0)):
            valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn,
                                             cls, tokenizer, step, opt.style)
            if eval_loss >= valid_loss:
                torch.save(model.state_dict(), 'checkpoints/{}_{}_{}_{}.chkpt'.format(
                    opt.model, opt.dataset, opt.order, opt.style))
                print('[Info] The checkpoint file has been updated.')
                eval_loss = valid_loss
                tab = 0
            else:
                tab += 1
            if tab == opt.patience:
                exit()

if __name__ == "__main__":
    main()
