import torch
import random
import numpy as np


def mask_word(w):
    _w_real = w
    _w_rand = np.random.randint(48265, size=w.shape)
    _w_mask = np.full(w.shape, 2)
    
    probs = torch.multinomial(torch.Tensor([0.8, 0.1, 0.1]), len(_w_real), replacement=True)

    _w = _w_mask * (probs == 0).numpy () + _w_real * (probs == 1).numpy () + _w_rand * (probs == 2).numpy ()
    return _w

def unfold_segments(segs):
    """Unfold the random mask segments, for example:
       The shuffle segment is [2, 0, 0, 2, 0],
       so the masked segment is like:
       [1, 1, 0, 0, 1, 1, 0]
       [1, 2, 3, 4, 5, 6, 7] (positions)
       (1 means this token will be masked, otherwise not)
       We return the position of the masked tokens like:
       [1, 2, 5, 6]
    """
    pos = []
    curr = 1  # We do not mask the start token
    for l in segs:
        if l >= 1:
            pos.extend ([curr + i for i in range (l)])
            curr += l
        else:
            curr += 1
    return np.array (pos)


def shuffle_segments(segs, unmasked_tokens):
    """
    We control 20% mask segment is at the start of sentences
               20% mask segment is at the end   of sentences
               60% mask segment is at random positions,
    """
    
    p = np.random.random ()
    if p >= 0.8:
        shuf_segs = segs[1:] + unmasked_tokens
    elif p >= 0.6:
        shuf_segs = segs[:-1] + unmasked_tokens
    else:
        shuf_segs = segs + unmasked_tokens
    
    random.shuffle (shuf_segs)
    
    if p >= 0.8:
        shuf_segs = segs[0:1] + shuf_segs
    elif p >= 0.6:
        shuf_segs = shuf_segs + segs[-1:]
    return shuf_segs


def get_segments(mask_len, span_len):
    segs = []
    while mask_len >= span_len:
        segs.append (span_len)
        mask_len -= span_len
    if mask_len != 0:
        segs.append (mask_len)
    return segs

def restricted_mask_sent(x, l, span_len=100000):
    """ Restricted mask sents
        if span_len is equal to 1, it can be viewed as
        discrete mask;
        if span_len -> inf, it can be viewed as
        pure sentence mask
    """
    if span_len <= 0:
        span_len = 1
    max_len = 0
    positions, inputs, targets, outputs, = [], [], [], []
    mask_len = round (len (x[:, 0]) * 0.5)
    len2 = [mask_len for i in range (l.size (0))]
    
    unmasked_tokens = [0 for i in range (l[0] - mask_len - 1)]
    segs = get_segments (mask_len, span_len)
    
    for i in range (l.size (0)):
        words = np.array (x[:l[i], i].tolist ())
        shuf_segs = shuffle_segments (segs, unmasked_tokens)

        pos_i = unfold_segments (shuf_segs)
        output_i = words[pos_i].copy ()
        target_i = words[pos_i - 1].copy ()
        words[pos_i] = mask_word (words[pos_i])
        inputs.append (words)
        targets.append (target_i)
        outputs.append (output_i)
        positions.append (pos_i - 1)
    
    x1 = torch.LongTensor (max (l), l.size (0)).fill_ (0)
    x2 = torch.LongTensor (mask_len, l.size (0)).fill_ (0)
    y = torch.LongTensor (mask_len, l.size (0)).fill_ (0)
    pos = torch.LongTensor (mask_len, l.size (0)).fill_ (0)
    l1 = l.clone ()
    l2 = torch.LongTensor (len2)
    for i in range (l.size (0)):
        x1[:l1[i], i].copy_ (torch.LongTensor (inputs[i]))
        x2[:l2[i], i].copy_ (torch.LongTensor (targets[i]))
        y[:l2[i], i].copy_ (torch.LongTensor (outputs[i]))
        pos[:l2[i], i].copy_ (torch.LongTensor (positions[i]))
    
    pred_mask = y != 0
    y = y.masked_select(pred_mask)
    print(x1.shape)
    print(l1.shape)
    print(x2.shape)
    print(l2.shape)
    print(y.shape)
    print(pred_mask.shape)
    print(pos.shape)
    return x1, l1, x2, l2, y, pred_mask, pos

# input_x = torch.LongTensor([[1, 1, 1],
#                             [3, 4, 5],
#                             [6, 7, 8],
#                             [9, 10, 11],
#                             [11, 12, 13],
#                             [14, 15, 16],
#                             [10, 1, 18],
#                             [12, 0, 66],
#                             [1, 0, 78],
#                             [0, 0, 1]])
# input_x = np.array([1, 28, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 18, 75, 68])
# print(mask_word(input_x))

input_x = torch.LongTensor([[1, 1, 1],
                            [3, 4, 5],
                            [6, 7, 8],
                            [9, 10, 11],
                            [11, 12, 13],
                            [14, 15, 16],
                            [10, 11, 18],
                            [12, 1, 66],
                            [1, 0, 78],
                            [0, 0, 1]])
lengths = torch.LongTensor([9, 7, 10])
restricted_mask_sent(input_x, lengths)