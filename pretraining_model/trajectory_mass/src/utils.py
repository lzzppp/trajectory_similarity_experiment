
import os
import re
import sys
import time
import pickle
import random
import inspect
import getpass
import argparse
import subprocess
import numpy as np
import torch
import logging
from torch import optim
from datetime import timedelta

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = '/checkpoint/%s/dumped' % getpass.getuser()
DYNAMIC_COEFF = ['lambda_clm', 'lambda_mlm', 'lambda_pc', 'lambda_ae', 'lambda_mt', 'lambda_bt', 'lambda_mass', 'lambda_bmt', 'lambda_span']

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def check_data_params(params):
    """
    Check datasets parameters.
    """
    # data path
    # assert os.path.isdir (params.data_path), params.data_path
    
    # check languages
    params.langs = params.lgs.split ('-') if params.lgs != 'debug' else ['en']
    assert len (params.langs) == len (set (params.langs)) >= 1
    # assert sorted(params.langs) == params.langs
    params.id2lang = {k: v for k, v in enumerate (sorted (params.langs))}
    params.lang2id = {k: v for v, k in params.id2lang.items ()}
    params.n_langs = len (params.langs)
    
    # CLM steps
    clm_steps = [s.split ('-') for s in params.clm_steps.split (',') if len (s) > 0]
    params.clm_steps = [(s[0], None) if len (s) == 1 else tuple (s) for s in clm_steps]
    assert all ([(l1 in params.langs) and (l2 in params.langs or l2 is None) for l1, l2 in params.clm_steps])
    assert len (params.clm_steps) == len (set (params.clm_steps))
    
    # MLM / TLM steps
    mlm_steps = [s.split ('-') for s in params.mlm_steps.split (',') if len (s) > 0]
    params.mlm_steps = [(s[0], None) if len (s) == 1 else tuple (s) for s in mlm_steps]
    assert all ([(l1 in params.langs) and (l2 in params.langs or l2 is None) for l1, l2 in params.mlm_steps])
    assert len (params.mlm_steps) == len (set (params.mlm_steps))
    
    # parallel classification steps
    params.pc_steps = [tuple (s.split ('-')) for s in params.pc_steps.split (',') if len (s) > 0]
    assert all ([len (x) == 2 for x in params.pc_steps])
    assert all ([l1 in params.langs and l2 in params.langs for l1, l2 in params.pc_steps])
    assert all ([l1 != l2 for l1, l2 in params.pc_steps])
    assert len (params.pc_steps) == len (set (params.pc_steps))
    
    # machine translation steps
    params.mt_steps = [tuple (s.split ('-')) for s in params.mt_steps.split (',') if len (s) > 0]
    assert all ([len (x) == 2 for x in params.mt_steps])
    assert all ([l1 in params.langs and l2 in params.langs for l1, l2 in params.mt_steps])
    assert all ([l1 != l2 for l1, l2 in params.mt_steps])
    assert len (params.mt_steps) == len (set (params.mt_steps))
    assert len (params.mt_steps) == 0 or not params.encoder_only
    
    # back machine translation steps
    params.bmt_steps = [tuple (s.split ('-')) for s in params.bmt_steps.split (',') if len (s) > 0]
    
    # denoising auto-encoder steps
    params.ae_steps = [s for s in params.ae_steps.split (',') if len (s) > 0]
    assert all ([lang in params.langs for lang in params.ae_steps])
    assert len (params.ae_steps) == len (set (params.ae_steps))
    assert len (params.ae_steps) == 0 or not params.encoder_only
    
    # mass steps
    params.mass_steps = [s for s in params.mass_steps.split (',') if len (s) > 0]
    mass_steps = []
    for src in params.mass_steps:
        for tgt in params.mass_steps:
            if src != tgt:
                mass_steps.append (tuple ([src, tgt]))
    
    # back-translation steps
    params.bt_steps = [tuple (s.split ('-')) for s in params.bt_steps.split (',') if len (s) > 0]
    assert all ([len (x) == 3 for x in params.bt_steps])
    assert all ([l1 in params.langs and l2 in params.langs and l3 in params.langs for l1, l2, l3 in params.bt_steps])
    assert all ([l1 == l3 and l1 != l2 for l1, l2, l3 in params.bt_steps])
    assert len (params.bt_steps) == len (set (params.bt_steps))
    assert len (params.bt_steps) == 0 or not params.encoder_only
    params.bt_src_langs = [l1 for l1, _, _ in params.bt_steps]
    
    # check monolingual datasets
    required_mono = set ([l1 for l1, l2 in (params.mlm_steps + params.clm_steps) if
                          l2 is None] + params.ae_steps + params.bt_src_langs + params.mass_steps)
    params.mono_dataset = {
        lang: {
            splt: os.path.join (params.data_path, '%s.%s.pth' % (splt, lang))
            for splt in ['train', 'valid', 'test']
        } for lang in params.langs if lang in required_mono
    }
    assert all ([all ([os.path.isfile (p) for p in paths.values ()]) for paths in params.mono_dataset.values ()])
    
    # check parallel datasets
    required_para_train = set (params.clm_steps + params.mlm_steps + params.pc_steps + params.mt_steps)
    required_para = required_para_train | set ([(l2, l3) for _, l2, l3 in params.bt_steps] + mass_steps)
    params.para_dataset = {
        (src, tgt): {
            splt: (os.path.join (params.data_path, '%s.%s-%s.%s.pth' % (splt, src, tgt, src)),
                   os.path.join (params.data_path, '%s.%s-%s.%s.pth' % (splt, src, tgt, tgt)))
            for splt in ['train', 'valid', 'test']
            if splt != 'train' or (src, tgt) in required_para_train or (tgt, src) in required_para_train
        } for src in params.langs for tgt in params.langs
        if src < tgt and ((src, tgt) in required_para or (tgt, src) in required_para)
    }
    assert all ([all ([os.path.isfile (p1) and os.path.isfile (p2) for p1, p2 in paths.values ()]) for paths in
                 params.para_dataset.values ()])
    
    # back parallel datasets
    params.back_dataset = {
        (src, tgt): (
            os.path.join (params.data_path, '%s-%s.%s.pth' % (src, tgt, src)),
            os.path.join (params.data_path, '%s-%s.%s.pth' % (src, tgt, tgt))
        ) for (src, tgt) in params.bmt_steps
    }
    
    # check that we can evaluate on BLEU
    assert params.eval_bleu is False or len (params.mt_steps + params.bt_steps + mass_steps) > 0
    
def get_optimizer(parameters, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)

class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        # update learning rate
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -0.5)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])

def BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def network_to_half(network):
    """
    Convert model to half precision in a batchnorm-safe way.
    """
    return BN_convert_float(network.half())

def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank > 0:
            filepath = '%s-%i' % (filepath, rank)
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path
    assert len(params.exp_name) > 0

    # create the sweep path if it does not exist
    sweep_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(sweep_path):
        subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()