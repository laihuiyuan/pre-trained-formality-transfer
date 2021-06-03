# -*- coding: utf-8 -*-

import torch

from torch import cuda
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

from utils.dataset import collate_fn
from transformers.modeling_bart import  make_padding_mask

device = 'cuda' if cuda.is_available() else 'cpu'


def optimize(opt, loss, retain_graph=False):
    opt.zero_grad()
    loss.backward(retain_graph=retain_graph)
    opt.step()


def cal_reward_loss(sample_probs, reward, idxs=None):
    sample_probs = sample_probs.contiguous()
    sample_logprobs = torch.log(sample_probs)
    reward = reward.unsqueeze(1).contiguous()
    if idxs is not None:
        batch_size, max_len = sample_probs.size()
        mask = torch.zeros(batch_size, max_len).to(device)
        for i, l in enumerate(idxs):
            mask[i, :l] = 1
        mask = mask.float().contiguous()
        output = -sample_logprobs * reward * mask
        output = (output.sum(-1)/mask.sum(-1)).mean()
    else:
        output = -sample_logprobs * reward
        output = output.mean()

    return output

def cal_bl_reward(inp, tgt):
    '''Caculate BLEU-based reward'''
    smooth = SmoothingFunction()
    bleus = []
    for hyp, ref in zip(inp, tgt):
        bleus.append(sentence_bleu([ref], hyp,
                                   smoothing_function=smooth.method1))
    bleus = torch.FloatTensor(bleus).to(device)

    return bleus

def cal_sc_loss(out, idx, cls, tokenizer, style):
    '''Caculate the loss of SC-based reward'''
    out = F.softmax(out, dim=-1)
    sample_probs, sample_idx = sample_3d(out)

    tgt = []
    for i, s in zip(idx.cpu(), sample_idx):
        e = torch.arange(len(s))[s.eq(tokenizer.eos_token_id)]
        e = e[0] if 0<len(e) and 4<e[0]<i else i-1
        tgt.append(s[:e].cpu().tolist())
    tgt_idx = collate_fn(tgt).to(device)
    tgt_cls = F.softmax(cls(tgt_idx).detach(),-1)

    if style == 0:
        tgt_reward = tgt_cls[:, 1] - tgt_cls[:, 0]
    else:
        tgt_reward = tgt_cls[:, 0] - tgt_cls[:, 1]

    loss_sc = cal_reward_loss(sample_probs, tgt_reward, idx)

    return loss_sc


def cal_bl_loss(out, tgt, idx, tokenizer):
    '''Caculate the loss of BLEU-based reward'''
    out = F.softmax(out, dim=-1)
    sample_probs, sample_idx = sample_3d(out)
    greedy_probs, greedy_idx = torch.max(out, dim=-1)

    tgt_sam, tgt_gre, tgt_ref = [], [], []
    for i, s, g, t in zip(idx.cpu(), sample_idx, greedy_idx, tgt):
        s_e = torch.arange(len(s))[s.eq(tokenizer.eos_token_id)]
        s_e = s_e[0] if 0<len(s_e) and 0<s_e[0]<i else i-1
        g_e = torch.arange(len(g))[g.eq(tokenizer.eos_token_id)]
        g_e = g_e[0] if 0<len(g_e) and 0<g_e[0]<i else i-1

        tgt_sam.append(s[:s_e].cpu().tolist())
        tgt_gre.append(g[:g_e].cpu().tolist())
        tgt_ref.append(t[1:i].cpu().tolist())

    tgt_sam = cal_bl_reward(tgt_sam, tgt_ref)
    tgt_gre = cal_bl_reward(tgt_gre, tgt_ref)
    loss_co = cal_reward_loss(sample_probs, (tgt_gre-tgt_sam)*0.2, idx)

    return loss_co


def sample_3d(probs, temperature=1):
    '''probs.shape = (batch, seq_len, dim)'''
    sample_idx = torch.zeros(probs.size(0), probs.size(1)).to(device)
    sample_probs = torch.zeros(probs.size(0), probs.size(1)).to(device)
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), temperature))
    else:
        temp = probs
    for i, s in enumerate(temp):
        temp_idx = torch.multinomial(s, 1)  # shape = (seq_len, 1)
        temp_probs = s.gather(1, temp_idx)  # shape = (seq_len, 1)
        sample_idx[i] = temp_idx.squeeze(1)
        sample_probs[i] = temp_probs.squeeze(1)

    return sample_probs, sample_idx.long()


def evaluate(model, valid_loader, loss_fn, 
             classifier, tokenizer, step, style):
    '''Evaluation function for BART'''
    model.eval()
    total_num = 0.
    total_acc = 0.
    total_loss = 0.
    with torch.no_grad():
        for batch in valid_loader:
            src, tgt = map(lambda x: x.to(device), batch)

            mask = make_padding_mask(src, tokenizer.pad_token_id)
            mask = 1 - mask.long() if mask is not None else None
            logits = model(src, attention_mask=mask,
                           decoder_input_ids=tgt)[0]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tgt[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
            probs, idxs = torch.max(logits, dim=-1)

            tgt = []
            for i in idxs:
                e = torch.arange(len(i))[i.eq(tokenizer.eos_token_id)]
                e = e[0] if 0<len(e) and e[0]<30 else 30
                tgt.append(i[:e].cpu().tolist())
            tgt = collate_fn(tgt).to(device)
            _, y_hat = torch.max(classifier(tgt),dim=-1)

            if style == 0:
                y_hat = [1 if p==1 else 0 for p in y_hat]
            else:
                y_hat = [1 if p==0 else 0 for p in y_hat]
            total_acc += sum(y_hat)
            total_num += len(tgt)
            total_loss += loss.mean()
    model.train()
    print('[Info] valid {:05d} | loss {:.4f} | acc_sc {:.4f}'.format(
        step, total_loss / len(valid_loader), total_acc / total_num))

    return total_loss / len(valid_loader), total_acc / total_num


def evaluate_sc(model, valid_loader, loss_fn, epoch):
    '''Evaluation function for style classifier'''
    model.eval()
    total_acc = 0.
    total_num = 0.
    total_loss = 0.
    with torch.no_grad():
        for batch in valid_loader:
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            total_loss += loss_fn(logits, y_batch)
            _, y_hat = torch.max(logits,dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)
    model.train()
    print('[Info] Epoch {:02d}-valid: {}'.format(
                epoch, 'acc {:.4f}% | loss {:.4f}').format(
        total_acc / total_num * 100, total_loss / total_num))

    return total_acc / total_num, total_loss / total_num
