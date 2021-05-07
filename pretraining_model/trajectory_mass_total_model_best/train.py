
import apex
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
# import torch.distributed as dist
from src.data_loader import DataLoader
from src.dataloader_trainer import EncoderwithDecoderTrainer
from src.slurm import init_distributed_mode, init_signal_handler
from src.utils import bool_flag, check_data_params, network_to_half, initialize_exp, get_optimizer
from src.model import check_model_params, build_model, transformer_raw_model

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
    parser.add_argument("--fp16", type=bool_flag, default=False,
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
    parser.add_argument("--clip_grad_norm", type=float, default=1,
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
    parser.add_argument("--lambda_mass_config", type=str, default="1",
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
    
    epochs = 100
    batch_size = 128
    gpus = [0, 1]
    parser = get_parser()
    params = parser.parse_args()

    # init_distributed_mode(params)
    logger = initialize_exp(params)
    # init_signal_handler()
    dataloader = DataLoader("/mnt/data4/lizepeng/porto/porto_grid_time_enough_dataset.pkl")
    
    check_data_params(params)
    check_model_params(params)

    params.n_words = 13416  # porto
    # params.n_words = 22339  # shanghai
    params.n_words_times = 1712  # porto
    params.pad_index = 0
    params.eos_index = 1
    params.mask_index = 2

    # traj_mass = build_model(params, [i for i in range(42609)])
    # encoder, decoder = build_model(params, [i for i in range(params.n_words)])
    # traj_mass = network_to_half(traj_mass)
    # traj_mass = apex.amp.initialize(traj_mass, opt_level="O1")
    # traj_mass.load_state_dict(torch.load("model0.pth"))
    
    model = transformer_raw_model.S2sTransformer(params.n_words, num_encoder_layers=params.n_layers).cuda()

    # traj_mass = apex.parallel.DistributedDataParallel(traj_mass, delay_allreduce=True)
    trainer = EncoderwithDecoderTrainer(model, params, data=dataloader, batch_size=batch_size)
    

    for epo in range(1, epochs):

        logger.info ("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_sentences = 0
        trainer.shuffle()
        
        # tloss = 0.0
        # while trainer.n_sentences < 1172843:
        trainer.mass_road_step(epo)
        trainer.bacth_size = batch_size

        # torch.save(trainer.encoder.state_dict(), "encoder" + str(epo) + ".pth")
        # torch.save(trainer.decoder.state_dict(), "decoder" + str(epo) + ".pth")
        # torch.save(trainer.encoder.state_dict(), "encoder" + str(epo) + ".pth")
        torch.save(trainer.model.state_dict(), "porto_model/porto_encoder_decoder_" + str(epo) + ".pth")

        trainer.end_epoch()
        logger.info("============ End of epoch %i ============" % trainer.epoch)
