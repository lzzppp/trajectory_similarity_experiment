
import re
import sys
import time
import math
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
from logging import getLogger
from src.optimizer import AdamInverseSqrtWithWarmup
from collections import OrderedDict
from src.utils import get_optimizer, update_lambdas
from src.dataset import myDataset
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
#from apex.fp16_utils import FP16_Optimizer
from torch.optim import Adam

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

    def __init__(self, model, params, data, batch_size, mask_pred=0):
        self.model = model
        # self.decoder = decoder
        self.params = params
        self.data = data.raw_dataset
        self.token_frequency = data.token_frequency_dict
        self.n_sentencess = 0
        self.mask_pred = mask_pred
        self.bacth_size = batch_size
        self.fp16 = False
        self.n_sentences = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.epoch = 0
        self.i = 0
        self.last_time = time.time()
        self.freq_weight = [1.0, 1.0, 1.0]
        for i in range(3, params.n_words):
            self.freq_weight.append(1.0/self.token_frequency[i])
        
        self.stats = OrderedDict(
            [('processed_s', 0), ('processed_w', 0)] +
            [('MA-%s' % lang, []) for lang in ["trajectory"]]
            # [('BT-%s-%s-%s' % (l1, l2, l3), []) for l1, l2, l3 in params.bt_steps]
        )

        self.MODEL_NAMES = ['encoder', 'decoder']
        
        # self.optimizers = {'model': self.get_optimizer_fp('model')}
        # 'model': self.get_optimizer_fp('model')
        # self.optimizers = {'encoder': self.get_optimizer_fp('encoder'),
        #                    'decoder': self.get_optimizer_fp('decoder')}
        # self.optimizer = Adam(self.model.parameters(), lr=0.0001)
        self.optimizer = AdamInverseSqrtWithWarmup(self.model.parameters())
                           # 'decoder': self.get_optimizer_fp('decoder')}
    
    def get_optimizer_fp(self, module):
        """
        Build optimizer.
        """
        assert module in ['model', 'encoder', 'decoder']
        optimizer = get_optimizer(getattr(self, module).parameters(), self.params.optimizer)
        if self.fp16:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        return optimizer
    
    def get_start_mask(self, length):
        mask_length = round(length * 0.5)
        # mask_length = 1
        # start = 1 # 开头引入eos 否则为0
        start = 0
        end = length-mask_length
        start_index = random.randint(start, end)
        return list(range(start_index, start_index+mask_length))
    
    def mask_word(self, w):
        _w_real = w
        _w_rand = np.random.randint(3, self.params.n_words, size=w.shape)
        _w_mask = np.full(w.shape, 2)
        
        probs = torch.multinomial(torch.Tensor([0.8, 0.1, 0.1]), len(_w_real), replacement=True)
        
        _w = _w_mask * (probs == 0).numpy() + _w_real * (probs == 1).numpy() + _w_rand * (probs == 2).numpy ()
        return _w
    
    def process_src_tgt_mask(self, src, tgt, mm, flag=0):
        if flag == 0:
            return src, tgt
        elif flag == 1:
            for i in range(0, len(src), 2):
                src[i] = True
            return src, tgt
        elif flag == 2:
            for i in range(1, len(src), 2):
                src[i] = True
            return src, tgt
        elif flag == 3:
            for i in range(0, len(src), 2):
                if mm[0] < i < mm[1]:
                    tgt[i - mm[0]] = True
            return src, tgt
        elif flag == 4:
            for i in range(1, len(src), 2):
                if mm[0] < i < mm[1]:
                    tgt[i - mm[0]] = True
            return src, tgt
        elif flag == 5:
            for i in range(0, len(src), 2):
                src[i] = True
            for i in range(1, len(src), 2):
                if mm[0] < i < mm[1]:
                    tgt[i - mm[0]] = True
            return src, tgt
        elif flag == 6:
            for i in range(1, len(src), 2):
                src[i] = True
            for i in range(0, len(src), 2):
                if mm[0] < i < mm[1]:
                    tgt[i - mm[0]] = True
            return src, tgt
    
    def produce_mask(self, mask_probs_, tgt_eps, src_masks, tgt_masks):
        # src_mask_, tgt_mask_ = [], []
        src_masks_new, tgt_masks_new = [], []
        for mask_prob, tgt_mm, src_mask_, tgt_mask_ in zip(mask_probs_, tgt_eps, src_masks, tgt_masks):
            src_mask_, tgt_mask_ = self.process_src_tgt_mask(src_mask_, tgt_mask_, tgt_mm, mask_prob)
            src_masks_new.append(src_mask_)
            tgt_masks_new.append(tgt_mask_)
        return src_masks_new, tgt_masks_new
    
    def get_batch(self):
        
        while self.i < len(self.data):
            batch_data_Raw, batch_data, batch_data2, lengths, length1, length2, position, pred_target, pred_mask, frequency_weight = [], [], [], [], [], [], [], [], [], []
            tgt_eps = []
            if self.i + self.bacth_size > len(self.data):
                self.bacth_size = len(self.data) - self.i
            for j in range(self.i, self.i + self.bacth_size):
                # batch_data_Raw = [1] + self.data[j][0] + [1] # 开头结尾添加eos
                # batch_data_example = [1] + self.data[j][0] + [1] # 开头结尾添加eos
                batch_data_Raw = [t for t in self.data[j][0]]
                batch_data_example = [w for w in self.data[j][0]]
                length = len(self.data[j][0])
                shuffle_token_list = self.get_start_mask(length)
                tgt_eps.append([shuffle_token_list[0], shuffle_token_list[-1]])
                position.append([pp + 1 for pp in shuffle_token_list])
                new_np_raw_list = np.array([batch_data_Raw[shuffle_index] for shuffle_index in shuffle_token_list])
                batch_data_mask = self.mask_word(new_np_raw_list)
                pred_target.extend([batch_data_Raw[shuffle_index] for shuffle_index in shuffle_token_list])
                for idx, shuffle_index in enumerate(shuffle_token_list):
                    batch_data_example[shuffle_index] = batch_data_mask[idx]
                batch_data.append(batch_data_example)
                batch_data2.append([2]+[batch_data_Raw[shuffle_index] for shuffle_index in shuffle_token_list][:-1])
                # length1.append(length + 2) # 开头结尾添加eos
                length1.append(length)
                length2.append(len(shuffle_token_list))
                # frequency_weight = [1.0/((self.token_frequency[target])*1.0) for target in pred_target]
                
            length1_max = max(length1)
            length2_max = max(length2)

            src_mask, tgt_mask, batch_data_token, batch_data2_token, batch_data_position, batch_data2_position = [], [], [], [], [], []
            for len1, len2, batch_data_line, batch_data2_line, shuffle_position in zip(length1, length2, batch_data, batch_data2, position):
                src_mask.append([False]*len1 + [True]*(length1_max - len1))
                tgt_mask.append([False]*len2 + [True]*(length2_max - len2))
                batch_data_token.append(batch_data_line + (length1_max - len1) * [0])
                batch_data2_token.append(batch_data2_line + (length2_max - len2) * [0])
                batch_data_position.append(list(range(1, length1_max + 1)))
                batch_data2_position.append(shuffle_position + [0] * (length2_max - len2))
            
            tgt_mask_raw = [[False]*len2 + [True]*(length2_max - len2) for len2 in length2]
            
            mask_probs = torch.multinomial(torch.Tensor([0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), self.bacth_size, replacement=True)
            src_mask, tgt_mask = self.produce_mask(mask_probs, tgt_eps, src_mask, tgt_mask)
            
            tgt_forward_mask = [[float(0.0)]*i + [float('-inf')]*(length2_max-i) for i in range(1, length2_max + 1)]
            
            self.i += self.bacth_size
            yield batch_data_token, batch_data2_token, batch_data_position, batch_data2_position, pred_target, src_mask, tgt_mask, tgt_mask_raw, tgt_forward_mask, length1

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
            # loss.backward()
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
    
    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        # update_lambdas(self.params, self.n_total_iter)
        # self.print_stats()
    
    def print_stats(self, pre_t_l):
        """
        Print statistics about the training.
        """
        if self.n_iter % 5 != 0:
            return

        pre10_total_loss = " - Pre10_total_loss = {:.4e}".format(pre_t_l)
        s_iter = "%7i - " % self.n_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # transformer learning rate
        lr = self.optimizer.param_groups[0]['lr']
        s_lr = " - Transformer LR = {:.4e}".format(lr)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} sent/s - {:8.2f} words/s - ".format(
            self.stats['processed_s'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_s'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr + pre10_total_loss)
    
    def shuffle(self):
        # print(self.data[0])
        random.shuffle(self.data)
    
    def end_epoch(self):
        """
        End the epoch.
        """
        self.epoch += 1
    
    def get_data_batch(self):
        x = next(self.data.get_batch())
        # self.data.shuffle_dataset()
        return x
    
    def mass_road_step(self, epoch):
        # self.encoder.train()
        # self.decoder.train()
        self.model.train()
        self.i = 0

        # progress = ProgressBar(len(self.data)//self.bacth_size, fmt=ProgressBar.FULL)
        tloss = 0.0
        loss_list = []
        f_weight = torch.FloatTensor(self.freq_weight).cuda()
        
        for b_d_token, b_d2_token, b_d_position, b_d2_position, p_target, src_mask, tgt_mask, tgt_mask_raw, tgt_f_mask, length_list in self.get_batch():
            b_d_token = torch.LongTensor(b_d_token).transpose(0, 1).cuda()
            b_d2_token = torch.LongTensor(b_d2_token).transpose(0, 1).cuda()
            b_d_position = torch.LongTensor(b_d_position).cuda()
            b_d2_position = torch.LongTensor(b_d2_position).cuda ()
            p_target = torch.LongTensor(p_target).cuda()
            src_mask = torch.BoolTensor(src_mask).cuda()
            tgt_mask = torch.BoolTensor(tgt_mask).cuda()
            tgt_mask_raw = torch.BoolTensor(tgt_mask_raw).cuda()
            tgt_f_mask = torch.FloatTensor(tgt_f_mask).cuda()
            
            pred_out = self.model(b_d_token,
                                  b_d2_token,
                                  b_d_position,
                                  b_d2_position,
                                  tgt_mask_raw,
                                  tgt_mask=tgt_f_mask, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
            
            loss = F.cross_entropy(pred_out, p_target)

            self.optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播
            self.optimizer.step()
            
            # loss = self.model(b_d_token, b_d2_token, b_d_position, p_target, l1, l2, p_t_mask)
            self.stats[('MA-%s' % "trajectory")].append(loss.item())

            # self.optimize(loss, 'model')
            # self.optimize(loss, ['model'])
            tloss += loss.cpu().detach().numpy()
            # progress.current += 1
            # progress()
            if int(self.n_sentences/self.bacth_size) % 10 == 0 and self.n_sentences > 0:
                # logger.info(tloss)
                # logger.info()
                self.print_stats(tloss)
                loss_list.append(tloss)
                tloss=0

            # number of processed sentences / words
            self.n_sentences += self.bacth_size
            self.stats['processed_s'] += b_d_token.shape[0]
            self.stats['processed_w'] += (np.array(length_list) - 1).sum().item()
            self.iter()

            del b_d_token, b_d2_token, b_d_position, b_d2_position, p_target, src_mask, tgt_mask, tgt_f_mask, loss, pred_out, tgt_mask_raw
            torch.cuda.empty_cache()

        pickle.dump(loss_list, open("loss_list_"+str(epoch)+".pkl", "wb"))
        # progress.done()
        self.shuffle()
