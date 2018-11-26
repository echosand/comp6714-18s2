
from config import config
import torch
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

_config = config()

# Task 1
def evaluate(golden_list, predict_list):
    if len(golden_list) != len(predict_list):
        return False
    p_denominator = 0
    r_denominator = 0
    right = 0
    pointer = 0
    while pointer < len(golden_list):
        if len(golden_list[pointer]) != len(predict_list[pointer]):
            return False
        else:
            tuple = evaluate_single(golden_list[pointer], predict_list[pointer])
            #print(tuple)
            p_denominator += tuple[0]
            r_denominator += tuple[1]
            right += tuple[2]
            pointer += 1
    if p_denominator ==0 and r_denominator == 0 and right == 0:
        f1=1
        return f1
    if p_denominator == 0:
        f1 = 0
        return f1
    if r_denominator == 0:
        f1 = 0
        return f1
    if right == 0:
        f1 = 0
        return  f1
    else:
        p=right/p_denominator
        r = right/r_denominator
        f1 = (2*p*r)/(p+r)
    #print(f1)
    return f1


def evaluate_single(golden_list, predict_list):
    if len(golden_list) != len(predict_list):
        return False
    gold = find_labels(golden_list)
    predict = find_labels(predict_list)
    r_denominator = len(gold)
    p_denominator = len(predict)
    right = 0
    pointer = 0
    while pointer < min(r_denominator, p_denominator):
        #print('gold predict', gold[pointer], predict[pointer])
        if gold[pointer] == predict[pointer]:
            right += 1
        pointer += 1
    return p_denominator, r_denominator, right


def find_labels(list):
    #print(list)
    if not list:
        return []
    if len(list) == 1:
        if list[0] == 'O':
            return []
        else:
            return [[list[0]]]
    if list[0] != 'O':
        new_list = [[list[0]]]
    else:
        new_list = []
    #print(new_list)
    for i in range(1, len(list)):
        if len(list[i]) > 1:
            if list[i][-3:] == list[i-1][-3:]:
                new_list[-1].append(list[i])
            else:
                new_list.append([list[i]])
    #print('new list',new_list)
    return new_list


# task 2
def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh) # caculation combination
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1) #seperate gates
    forgetgate = torch.sigmoid(forgetgate)
    ingate = 1 - forgetgate
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy

# Task 3
def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
    embedding= model.char_embeds
    char_lstm_embeds=embedding(batch_char_index_matrices)
    size= char_lstm_embeds.size()
    char_lstm_embeds = char_lstm_embeds.view(size[0]*size[1],size[2],size[3])
    batch_word_len_lists = batch_word_len_lists.view(batch_word_len_lists.size()[0] * batch_word_len_lists.size()[1])
    # batch_word size from (7*10) to 70
    perm_idx, sorted_batch_word_len_list = model.sort_input(batch_word_len_lists)
    sorted_input_embeds =char_lstm_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)
    sorted_batch_word_len_list: object
    output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_list.data.tolist(),
                                           batch_first=True)
    output_sequence, (hn,cn) = model.char_lstm(output_sequence)
    raw = hn[0][desorted_indices]
    forward1 = raw[:len(raw) // 2]
    backward1 = raw[len(raw) // 2:]
    raw2 = hn[1][desorted_indices]
    forward2 = raw2[:len(raw2) // 2]
    backward2 = raw2[len(raw2) // 2:]
    output1=torch.cat((forward1,forward2),dim=-1)
    output2=torch.cat((backward1,backward2),dim=-1)
    output = torch.cat((output1,output2),dim=0)
    output = output.view(size[0],size[1],2*model.config.char_lstm_output_dim)
    return output




