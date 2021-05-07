
import re
import sys
import math
import torch
import pickle
import random
import numpy as np
from logging import getLogger
from collections import OrderedDict
from src.utils import get_optimizer
from torch.nn.utils import clip_grad_norm_
from apex.fp16_utils import FP16_Optimizer

logger = getLogger()


class ProgressBar(object):
    DEFAULT='Progress: %(bar)s %(percent)3d%%'
    FULL='%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total,width = 40,fmt = DEFAULT,symbol = '=',
                 output=sys.stderr):
        assert len(symbol) == 1
        self.total=total
        self.width=width
        self.symbol=symbol
        self.output=output
        self.fmt=re.sub(r'(?P<name>%\(.+?\))d', r'\g<name>%dd' % len(str(total)), fmt)
        self.current=0

    def __call__(self):
        percent=self.current/float(self.total)
        size=int(self.width*percent)
        remaining=self.total-self.current
        bar='[' + self.symbol * size + ' ' * (self.width-size) + ']'

        args={
            'total':self.total,
            'bar':bar,
            'current':self.current,
            'percent':percent*100,
            'remaining':remaining}
        
        print ('\r'+self.fmt%args,file=self.output,end = '')

    def done(self):
        self.current=self.total
        self()
        print('', file=self.output)

class EncoderwithDecoderTrainer(object):

    def __init__(self, traj_model, params, data, batch_size, mask_pred=0):
        self.model = traj_model
        # self.decoder = decoder
        self.params = params
        self.data = data.raw_dataset
        self.mask_pred = mask_pred
        self.bacth_size = batch_size
        self.fp16 = True
        self.n_sentences = 0

        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            # [('CLM-%s' % l, []) for l in params.langs] +
            # [('CLM-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys ()] +
            # [('CLM-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys ()] +
            # [('MLM-%s' % l, []) for l in params.langs] +
            # [('MLM-%s-%s' % (l1, l2), []) for l1, l2 in data['para'].keys ()] +
            # [('MLM-%s-%s' % (l2, l1), []) for l1, l2 in data['para'].keys ()] +
            # [('PC-%s-%s' % (l1, l2), []) for l1, l2 in params.pc_steps] +
            # [('AE-%s' % lang, []) for lang in params.ae_steps] +
            # [('MT-%s-%s' % (l1, l2), []) for l1, l2 in params.mt_steps] +
            # [('BMT-%s-%s' % (l1, l2), []) for l1, l2 in params.bmt_steps] +
            [('MA-%s' % lang, []) for lang in ["trajectory"]]
            # [('BT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps]
        )

        self.optimizers = {
            'model': self.get_optimizer_fp('model')
        }
    
    def get_optimizer_fp(self, module):
        """
        Build optimizer.
        """
        assert module in ['model', 'encoder', 'decoder']
        optimizer = get_optimizer(getattr(self, module).parameters (), self.params.optimizer)
        if self.fp16:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        return optimizer
    
    def get_start_mask(self, length):
        mask_length = round(length / 2)
        start = 1
        end = length-mask_length
        start_index = random.randint(start, end)
        return list(range(start_index, start_index+mask_length+1))
    
    def mask_word(self, w):
        _w_real = w
        _w_rand = np.random.randint(3, 42899, size=w.shape)
        _w_mask = np.full(w.shape, 2)
        
        probs = torch.multinomial(torch.Tensor([0.8, 0.1, 0.1]), len(_w_real), replacement=True)
        
        _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy ()
        return _w
    
    def get_batch(self):
        i=0
        while i < len(self.data):
            batch_data_Raw, batch_data, batch_data2, lengths, length1, length2, position, pred_target, pred_mask = [], [], [], [], [], [], [], [], []
            if i + self.bacth_size > len(self.data):
                self.bacth_size = len(self.data) - i
            for j in range(i, i + self.bacth_size):
                batch_data_Raw.append([[1] + self.data[j][0] + [1], [0] + self.data[j][1] + [self.data[j][1][-1] + 15]])
                batch_data.append([[1] + self.data[j][0] + [1], [0] + self.data[j][1] + [self.data[j][1][-1] + 15]])
                lengths.append(len(self.data[j][0]))
                length1.append(len(self.data[j][0]) + 2)
            for length_idx in range(len(lengths)):
                length = lengths[length_idx]
                shuffle_token_list = self.get_start_mask(length)
                position.append(shuffle_token_list)
                batch_data_mask = self.mask_word(np.array([batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list]))
                pred_target.extend([batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list])
                for idx, shuffle_index in enumerate(shuffle_token_list):
                    batch_data[length_idx][0][shuffle_index] = batch_data_mask[idx]
                batch_data2.append([[2] + [batch_data_Raw[length_idx][0][shuffle_index] for shuffle_index in shuffle_token_list[:-1]],
                                    [batch_data_Raw[length_idx][1][shuffle_index] for shuffle_index in shuffle_token_list]])
                length2.append(len(shuffle_token_list))
            length1_max = max(length1)
            length2_max = max(length2)
            pred_mask = [l2*[True] + (length2_max-l2)*[False] for l2 in length2]
            batch_data_token = [line[0] + (length1_max-len(line[0]))*[0] for line in batch_data]
            batch_data_time = [line[1] + (length1_max-len(line[1]))*[0] for line in batch_data]
            batch_data2_token = [line[0] + (length2_max-len(line[0]))*[0] for line in batch_data2]
            batch_data2_time = [line[1] + (length2_max-len(line[1]))*[0] for line in batch_data2]
            batch_data_position = [shuffle_position + [510]*(length2_max - len(shuffle_position)) for shuffle_position in position]
            i += self.bacth_size
            yield batch_data_token, batch_data_time, batch_data2_token, batch_data2_time, batch_data_position, pred_target, length1, length2, pred_mask
    
    def optimize(self, loss, modules):
        """
        Optimize.
        """
        if type(modules) is str:
            modules = [modules]

        # check NaN
        if (loss != loss).data.any():
            logger.error("NaN detected")
            exit()

        # zero grad
        for module in modules:
            self.optimizers[module].zero_grad()

        # backward
        if self.params.fp16:
            assert len(modules) == 1, "fp16 not implemented for more than one module"
            self.optimizers[module].backward(loss)
        else:
            loss.backward()

        # clip gradients
        if self.params.clip_grad_norm > 0:
            for module in modules:
                if self.params.fp16:
                    self.optimizers[module].clip_master_grads(self.params.clip_grad_norm)
                else:
                    clip_grad_norm_(getattr(self, module).parameters(), self.params.clip_grad_norm)

        # optimization step
        for module in modules:
            self.optimizers[module].step()
    
    def shuffle(self):
        random.shuffle(self.data)
    
    def mass_road_step(self, epoch):
        self.model.train()

        progress = ProgressBar(len(self.data)//self.bacth_size, fmt=ProgressBar.FULL)
        tloss = 0.0
        loss_list = []
        
        for b_d_token, b_d_time, b_d2_token, b_d2_time, b_d_position, p_target, l1, l2, p_t_mask in self.get_batch():
            b_d_token = torch.LongTensor(b_d_token).cuda()
            b_d_time = torch.LongTensor(b_d_time).cuda()
            b_d2_token = torch.LongTensor(b_d2_token).cuda()
            b_d2_time = torch.LongTensor(b_d2_time).cuda()
            b_d_position = torch.LongTensor(b_d_position).cuda()
            p_target = torch.LongTensor(p_target).cuda()
            l1=torch.LongTensor(l1).cuda()
            l2=torch.LongTensor(l2).cuda()
            p_t_mask=torch.ByteTensor(p_t_mask).transpose(0, 1).cuda()
            
            # b_d_token = torch.LongTensor(b_d_token)
            # b_d_time = torch.LongTensor(b_d_time)
            # b_d2_token = torch.LongTensor(b_d2_token)
            # b_d2_time = torch.LongTensor(b_d2_time)
            # b_d_position = torch.LongTensor(b_d_position)
            # p_target = torch.LongTensor(p_target)
            # l1=torch.LongTensor(l1)
            # l2=torch.LongTensor(l2)
            # p_t_mask=torch.ByteTensor(p_t_mask).transpose(0, 1)
            
            # loss = self.model(b_d_token, b_d_time, b_d2_token, b_d2_time, b_d_position, p_target, l1, l2, p_t_mask)
            # print(b_d_token.shape, b_d2_token.shape, b_d_position.shape, p_target.shape, l1.max(), l2.shape, p_t_mask.shape)
            loss = self.model(b_d_token, b_d2_token, b_d_position, p_target, l1, l2, p_t_mask)
            
            # enc1 = self.encoder('fwd', xy=[b_d_token, b_d_time], lengths=l1, causal=False)
            # enc1 = enc1.transpose(0, 1)
            #
            # enc_mask = b_d_token.ne(self.params.mask_index)
            # enc_mask = enc_mask.transpose(0, 1)
            #
            # dec2 = self.decoder('fwd',
            #                     xy=[b_d2_token, b_d2_time], lengths=l2, causal=True,
            #                     src_enc=enc1, src_len=l1, positions=b_d_position.transpose(1, 0), enc_mask=enc_mask)
            #
            # _, loss = self.decoder('predict', tensor=dec2, pred_mask=p_t_mask, y=p_target, get_scores=False)
            self.stats[('MA-%s' % "trajectory")].append(loss.item())
    
            self.optimize(loss, 'model')
            tloss += loss
            progress.current += 1
            progress()
            if progress.current%1000 == 0:
                print(tloss)
                loss_list.append(tloss)
                tloss=0
            
            # number of processed sentences / words
            self.n_sentences += self.bacth_size
            self.stats['processed_s'] += l2.size(0)
            self.stats['processed_w'] += (l2 - 1).sum ().item ()
        pickle.dump(loss_list, open("loss_list_"+epoch+".pkl", "wb"))
        progress.done()
        self.shuffle()
