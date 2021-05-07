
import apex
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from src.data_loader import DataLoader
from src.trainer import EncoderwithDecoderTrainer
from src.slurm import init_distributed_mode, init_signal_handler
from src.utils import bool_flag, check_data_params, network_to_half, initialize_exp
from src.model import check_model_params, build_model

###################################################################
# sentences = [[2, 2, 3, 4, 5, 6, 7], [4, 5, 6, 8, 7], [16, 5, 17, 81, 88, 91]]
#
# lengths = torch.LongTensor ([len (s) + 2 for s in sentences])
# sent = torch.LongTensor (lengths.max ().item (), lengths.size (0)).fill_(0)
#
# sent[0] = 1
# for i, s in enumerate(sentences):
#     if lengths[i] > 2:  # if sentence not empty
#         sent[1:lengths[i] - 1, i].copy_ (torch.from_numpy (np.array(s)))
#     sent[lengths[i] - 1, i] = 1
# print(sent.shape)
###################

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16
    parser.add_argument("--fp16", type=bool_flag, default=True,
                        help="Run model with float16")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=False,
                        help="Only use an encoder")
    parser.add_argument("--english_only", type=bool_flag, default=False,
                        help="Only use english domain (equal to only use one language)")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=128,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_dec_layers", type=int, default=6,
                        help="Number of Decoder Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--attention_setting", type=str, default="v1", choices=["v1", "v2"],
                        help="Setting for attention module, benefits for distinguish language")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")
    parser.add_argument("--word_mass", type=float, default=0,
                        help="Randomly mask input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--min_len", type=int, default=0,
                        help="Minimum length of sentences (after BPE)")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_bmt", type=str, default="1",
                        help="Back Parallel coefficient")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")
    parser.add_argument("--lambda_mass", type=str, default="1",
                        help="MASS coefficient")
    parser.add_argument("--lambda_span", type=str, default="10000",
                        help="Span coefficient")

    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--bmt_steps", type=str, default="",
                        help="Back Machine Translation step")
    parser.add_argument("--mass_steps", type=str, default="",
                        help="MASS prediction steps")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")

    # reload a pretrained model
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    return parser

if __name__ == "__main__":
    
    epochs = 10
    parser = get_parser()
    params = parser.parse_args()

    # init_distributed_mode(params)
    logger = initialize_exp(params)
    # init_signal_handler()
    dataloader = DataLoader("src/data/trajectory_new_new_grid_dataset.pkl")
    
    check_data_params(params)
    check_model_params(params)

    params.n_words = 42899  # porto
    params.pad_index = 0
    params.eos_index = 1
    params.mask_index = 2

    traj_mass = build_model(params, [i for i in range(42899)])

    # encoder = network_to_half(encoder)
    # decoder = network_to_half(decoder)
    traj_mass = network_to_half(traj_mass)

    # traj_mass = apex.parallel.DistributedDataParallel(traj_mass, delay_allreduce=True)
    # decoder = apex.parallel.DistributedDataParallel(decoder, delay_allreduce=True)
    
    # encoder = nn.DataParallel(encoder)
    # decoder = nn.DataParallel(decoder)

    # encoder = encoder.cuda()
    # decoder = decoder.cuda()
    
    # encoder = encoder.cpu()
    # decoder = decoder.cpu()

    # b_d_token = torch.randint(0, 20, [3, 10])
    # b_d_time = torch.randint(0, 20, [3, 10])
    # b_d2_token = torch.randint(0, 20, [3, 5])
    # b_d2_time = torch.randint(0, 20, [3, 5])
    # b_d_position = torch.randint(1, 20, [3, 5])
    # p_target = torch.randint(1, 20, [12])
    # l1 = torch.randint(1, 10, [3])
    # l2 = torch.randint(1, 5, [3])
    # p_t_mask = torch.BoolTensor([[True, True, True],
    #                             [True, True, True],
    #                             [True, True, True],
    #                             [True, True, False],
    #                             [True, False, False]])
    # enc1 = encoder('fwd', xy=[b_d_token.transpose(1, 0), b_d_time.transpose(1, 0)], lengths=l1, causal=False)
    # enc1 = enc1.transpose(0, 1)
    #
    # #
    # enc_mask = b_d_token.transpose(0, 1).ne(2)
    # enc_mask = enc_mask.transpose(0, 1)
    # #
    # dec2 = decoder('fwd',
    #                xy=[b_d2_token.transpose(1, 0), b_d2_time.transpose(1, 0)], lengths=l2, causal=True,
    #                src_enc=enc1, src_len=l1, positions=b_d_position.transpose(1, 0), enc_mask=enc_mask)
    # print(dec2.shape)
    # #
    # _, loss = decoder('predict', tensor=dec2, pred_mask=p_t_mask, y=p_target, get_scores=False)
    #
    # input_x = torch.LongTensor([[1, 1, 1],
    #                             [3, 4, 5],
    #                             [6, 7, 8],
    #                             [9, 10, 11],
    #                             [11, 12, 13],
    #                             [14, 15, 16],
    #                             [10, 1, 18],
    #                             [12, 0, 66],
    #                             [1, 0, 78],
    #                             [0, 0, 1]]).cuda()
    # lengths = torch.LongTensor([9, 7, 10]).cuda()
    # enc1 = encoder('fwd', x=input_x, lengths=lengths, causal=False)
    #
    trainer = EncoderwithDecoderTrainer(traj_mass, params, data=dataloader, batch_size=16)
    for epoch in range(10):
        trainer.mass_road_step(epoch + 1)
        torch.save(trainer.model.state_dict(), "model.pth")
        # torch.save(trainer.decoder.state_dict(), "decoder.pth")
