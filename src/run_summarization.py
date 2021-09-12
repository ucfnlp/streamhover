import argparse
import os
import random
import numpy as np
import time
import warnings
from tqdm import tqdm
import pandas as pd
import pickle
from collections import defaultdict, Counter
import logging

import pdb
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data

from data import build_data_iterator, BehanceClipDataset
from modeling_bertvqvae import VQVAESum, build_optim, build_optim_bert, build_optim_not_bert

from models.loss import NMTLossCompute  # , abs_loss
from models.trainer import build_trainer

from transformers import BertTokenizer
from utils.logging import init_logger, logger
from utils.utils import calc_prf_rouge_bertscore, print_stats_int, create_dir, save_json


model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'enc_layers', 'enc_hidden_size', 'enc_ff_size',
               'dec_layers', 'dec_hidden_size', 'dec_ff_size', 'encoder', 'ff_actv', 'use_interval']


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser(description='VQ-VAE Summarization')

    # general
    parser.add_argument('--seed', default=None, type=int, help='random seed for initializing training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
    parser.add_argument('--visible_gpus', default='-1', type=str)
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:8880', type=str, help='url used to set up distributed training')

    parser.add_argument('--mode', default='train', type=str, choices=['train', 'val', 'test', 'val_grid', 'test_grid', 'inference_train', 'inference_val', 'inference_test', 'video'],
                        help="'inference' is used to generate a summary for one clip; 'video' mode will generate summaries for every 5 min. clips")
    parser.add_argument('--data_path', default='../data/behance')
    parser.add_argument('--model_path', default='../models/')
    parser.add_argument('--result_path', default='../results/')
    parser.add_argument('--temp_dir', default='../temp')
    parser.add_argument('--log_file', default='../logs/behance.log')

    # train
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--train_from', default='', type=str, metavar='PATH', help='trained model path')
    parser.add_argument('--accum_count', default=5, type=int, help='number of accumulated loss')
    parser.add_argument('--report_every', default=100, type=int, help='output every this setting')
    parser.add_argument('--save_checkpoint_steps', default=5, type=int, help='save trained model at every this setting')

    parser.add_argument('--debug', action='store_true')

    # test, validate
    parser.add_argument('--video_inference_id', default=7, type=int, help='video id for video-wise inference; summaries generated for every 5-min. clips; 0 - 2359')
    parser.add_argument('--video_inf_min_sent', default=30, type=int, help='minimum number of sentences in 5-min. clip to do inference; clips with less than this number will be skipped')
    parser.add_argument('--inference_cid', default=0, type=int, help='inference for a given clip id')
    parser.add_argument('--num_sum_sent', default=5, nargs='+', type=int, help='num. of summary sentences per clip')
    parser.add_argument('--input_dist_level', default='video', choices=['video', 'clip'], help='different levels of input utterances to compute distance to codebook')
    parser.add_argument('--is_sample_all', action='store_true', help='set True if all candidate sentences are used for voting')
    parser.add_argument('--topn_exclude', default=2, type=int, help='exclude representations assigned to top-n popular clusters for voting; 1: top-1, 2: top-1,2')
    parser.add_argument('--num_min_word', default=5, type=int, help='sentences with this setting of words will not be used')
    parser.add_argument('--ntop_cluster', default=50, type=int, help='num. top-n popular clusters used for summarization voting; best param hard coded')
    parser.add_argument('--ntop_sent', default=20, type=int, help="num. top-n sentence sampling for voting; not used when '--is_sample_all' is activated; best param hard coded")
    parser.add_argument('--ext_sum_ref', action='store_true', help='set True if human reference is extractive summaries')

    parser.add_argument('--report_rouge', action='store_false')
    parser.add_argument('--rouge_mean', action='store_true', help='used for significance test; default set: generate rouge score for each test sample')
    parser.add_argument('--force', action='store_true')

    # model
    parser.add_argument('--ch_dim', type=int, default=100, help='D filters in conv. encoder and decoder')
    # VQ-VAE
    parser.add_argument('--num_cluster', type=int, default=1024, help='num. of codebook')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='beta scalar')
    parser.add_argument('--VQVAE_decay', type=float, default=0.99)
    parser.add_argument('--max_pos', default=512, type=int, help='max. length of input')
    parser.add_argument('--num_avg_sample', default=30, type=int, help='num. of samples for multi-nomial sampling; not used')
    parser.add_argument('--skip_VQ', action='store_true', help='set True if you want to train without VQ-VAE; train encoder-decoder only')
    # decoder
    parser.add_argument('--dec_dropout', default=0.2, type=float)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dec_hidden_size', default=768, type=int)
    parser.add_argument('--dec_heads', default=8, type=int)
    parser.add_argument('--dec_ff_size', default=2048, type=int)
    # loss
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='0 for softmax, >0 for KL-divergence')

    # optimizer
    parser.add_argument('--optim_no_load_checkpoint', action='store_true', help='set True if an optimizer starts from scratch')
    parser.add_argument('--search_model_name', default='bert.model', choices=['bert.model', 'vq_vae'], help='target sub-model to set different LR')
    parser.add_argument('--sep_optim', action='store_true', help='set to True if different LRs are used')
    parser.add_argument('--lr_bert', default=7e-4, type=float, help='LR for BERT(encoder) or other target model')
    parser.add_argument('--lr_not_bert', default=4e-2, type=float, help='LR for rest')
    parser.add_argument('--warmup_steps_bert', default=25000, type=int, help='warmup steps for BERT(encoder)')
    parser.add_argument('--warmup_steps_not_bert', default=15000, type=int, help='warmup steps for rest')
    parser.add_argument('--lr', default=0.25, type=float, help='LR if one LR is used')
    parser.add_argument('--warmup_steps', default=15000, type=int, help='warmup steps')

    parser.add_argument('--optim', default='adam', type=str)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--max_grad_norm', default=0, type=float)
    parser.add_argument('--decay_steps', default=1000, type=int)

    args = parser.parse_args()

    args.model_path = '{}_c{}_e{}/'.format(args.model_path, args.num_cluster, args.ch_dim)
    create_dir(args.model_path)
    create_dir(args.result_path)
    if not os.path.isdir(args.log_file):
        log_dir = os.path.dirname(args.log_file)
        create_dir(log_dir)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    init_logger(args.log_file)
    logger.info(args)
    args.gpu = gpu
    args.n_gpu = ngpus_per_node
    args.visible_gpus = ngpus_per_node

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=1, rank=0)

    # load checkpoint
    if args.train_from != '':
        if os.path.isfile(args.train_from):
            logger.info('Loading checkpoint from %s' % args.train_from)
            if args.gpu is None:
                checkpoint = torch.load(args.train_from)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.train_from, map_location=loc)
                opt = vars(checkpoint['opt'])
                for k in opt.keys():
                    if (k in model_flags):
                        setattr(args, k, opt[k])
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.train_from))
    else:
        checkpoint = None

    # create model
    logger.info('creating model')
    model = VQVAESum(args)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'],
               'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'],
               'EOQ': tokenizer.vocab['[unused2]']}

    # loss
    recon_loss = NMTLossCompute(model.generator, symbols, model.vocab_size, label_smoothing=args.label_smoothing)
    cuda_dev = 'cuda' if args.gpu is None else 'cuda:{}'.format(args.gpu)
    device = torch.device(cuda_dev)  # 'cuda'
    recon_loss.to(device)

    # optimizer
    checkpoint_optim = None if args.optim_no_load_checkpoint else checkpoint
    if args.sep_optim:
        optim_bert = build_optim_bert(args, model, checkpoint_optim)
        optim_not_bert = build_optim_not_bert(args, model, checkpoint_optim)
        optimizer = [optim_bert, optim_not_bert]
    else:
        optimizer = [build_optim(args, model, checkpoint_optim)]

    # model
    batch_size_total = args.batch_size * args.accum_count
    if not torch.cuda.is_available():
        logger.info('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # load parameter to model
    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch'] if args.mode == 'train' else 0

        model_dict = model.state_dict()
        pretrain_model_dict = checkpoint['model']
        if args.gpu is None:
            pretrained_dict = {k: v for k, v in pretrain_model_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        else:
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrain_model_dict.items() if k.replace('module.', '') in model_dict and v.shape == model_dict[k.replace('module.', '')].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        logger.info("=> loaded checkpoint '{}' [dict size:{}] (epoch {})".format(args.train_from, len(pretrained_dict), checkpoint['epoch']))

    cudnn.benchmark = True

    if args.mode == 'train':
        # Data loading
        data_loader, data_sampler = build_data_iterator(args, tokenizer)

        # train
        num_train_data = len(data_loader.dataset)
        num_total_steps = (num_train_data // batch_size_total) * args.epochs
        step = optimizer[0]._step + 1
        logger.info('num_train_data: {}'.format(num_train_data))
        logger.info('num_total_steps: {}'.format(num_total_steps))
        logger.info('num_total_steps accumulated: {}'.format(step + num_total_steps))

        trainer = build_trainer(args, model, optimizer, recon_loss, step + num_total_steps)

        for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
            trainer.train(data_loader, epoch)

    elif 'inference' in args.mode:
        # inference for one clip in test or val or train
        # video level distance computing is not supported
        trainer = build_trainer(args, model, optimizer, recon_loss)
        clip_data = BehanceClipDataset(args.data_path, args.mode, args.num_min_word)

        # best parameter setting for voting
        if args.is_sample_all:
            nc = 50
            ns = 10     # not used but just for consistency
        else:
            nc = 60
            ns = 20

        if 'train' in args.mode and args.inference_cid >= 3884:
            print ('The given clip id ({}) exceed the total num. of clips(3884) exist in train data - set to the last one (3883)'.format(args.inference_cid))
            inference_split = 'train'
            args.inference_cid = 3883
        if 'val' in args.mode and args.inference_cid >= 728:
            print ('The given clip id ({}) exceed the total num. of clips(728) exist in validation data - set to the last one (727)'.format(args.inference_cid))
            inference_split = 'val'
            args.inference_cid = 727
        if 'test' in args.mode and args.inference_cid >= 809:
            print ('The given clip id ({}) exceed the total num. of clips(809) exist in test data - set to the last one (808)'.format(args.inference_cid))
            inference_split = 'test'
            args.inference_cid = 808

        data_loader, data_sampler = build_data_iterator(args, tokenizer, clip_ids=[args.inference_cid])
        distances = trainer.eval_dist(data_loader)

        pred_id, vote_rate = gen_summary(distances, clip_data, [args.inference_cid], nc, ns, args.topn_exclude, args.is_sample_all, args.debug)
        transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url, valid_sent_id = clip_data[args.inference_cid]

        n_sum = args.num_sum_sent[0]
        pred_id = sorted(pred_id[0][0][:n_sum])
        pred = [transcript[pid]['display'].strip() for pid in pred_id]
        pred_time = [transcript[pid]['offset'] for pid in pred_id]

        # save summary in a json file
        data_json = {}
        data_json['summary'] = []
        data_json['video_url'] = vid_url
        for pi in range(len(pred)):
            data_json['summary'].append({'utterance': pred[pi], 'start_time': pred_time[pi]})
        save_json(data_json, os.path.join(args.result_path, 'summary_{}_cid{}.json'.format(args.mode, args.inference_cid)))

    elif args.mode == 'video':
        # inference video-wide summaries for every 5-min. clips
        # video level distance computing is not supported
        trainer = build_trainer(args, model, optimizer, recon_loss)
        clip_data = BehanceClipDataset(args.data_path, args.mode, args.num_min_word)

        # best parameter setting
        args.is_sample_all = True
        nc = 50
        ns = 10

        vid_2clip = defaultdict(list)
        vid_set = set()
        for ci, cdata in enumerate(clip_data):
            transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url, valid_sent_id = cdata
            vid_2clip[video_id].append(ci)
            vid_set.add(video_id)

        idx_2vid = {vi: vid for vi, vid in enumerate(vid_set)}
        vid_id = idx_2vid[args.video_inference_id]
        clips = vid_2clip[vid_id]
        n_sum = args.num_sum_sent[0]

        pred, pred_time, pred_dur = [], [], []
        transcript_video = []
        n_valid_clip = 0
        for cid in tqdm(sorted(clips)):
            transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url, valid_sent_id = clip_data[cid]
            if len(transcript) < args.video_inf_min_sent:
                logger.info('clip id {} is skipped: num. utterance = {} < {}'.format(cid, len(transcript), args.video_inf_min_sent))
                continue

            n_valid_clip += 1
            data_loader, data_sampler = build_data_iterator(args, tokenizer, clip_ids=[cid])
            distances = trainer.eval_dist(data_loader)

            pred_id, vote_rate = gen_summary(distances, clip_data, [cid], nc, ns, args.topn_exclude, args.is_sample_all, args.debug)

            pred_id = sorted(pred_id[0][0][:n_sum])

            pred_cur = [transcript[pid]['display'].strip() for pid in pred_id]
            pred_time_cur = [transcript[pid]['offset'] for pid in pred_id]
            pred_time_dur = [transcript[pid]['duration'] for pid in pred_id]

            pred += pred_cur
            pred_time += pred_time_cur
            pred_dur += pred_time_dur

            transcript_video.extend(transcript)

        logger.info('vid:{} ({} clips) - {} summary senteces are generated from each {} clip'.format(vid_id, len(clips), n_sum, n_valid_clip))

        # save summary in a json file
        data_json = {}
        data_json['summary'] = []
        data_json['video_url'] = vid_url
        for pi in range(len(pred)):
            data_json['summary'].append({'utterance': pred[pi], 'start_time': pred_time[pi], 'duration': pred_dur[pi]})
        save_json(data_json, os.path.join(args.result_path, 'summary_vid{}.json'.format(vid_id)))

        data_trans = {}
        data_trans['transcript'] = transcript_video
        save_json(data_trans, os.path.join(args.result_path, 'transcript_vid{}.json'.format(vid_id)))

    else:   # if 'grid' in args.mode:
        # summaries will be stored in the $results directory
        # outputs a ().csv file that has all result scores (Rouge, P,R,F1, Bert-score) for different parameters
        start_time = time.time()
        trainer = build_trainer(args, model, optimizer, recon_loss)
        clip_data = BehanceClipDataset(args.data_path, args.mode, args.num_min_word)
        logger.info('{}: {} clips'.format(args.mode, len(clip_data)))
        data_len = len(clip_data)

        is_sample_all = args.is_sample_all
        if args.debug:
            nc_range = [20]
            ns_range = [10]
        else:
            if 'grid' in args.mode:
                # grid search to find the best value of top-N cluster & top-M sampling
                if is_sample_all:
                    nc_range = list(range(10, 71, 10))   # [30, 40, 50, 60, 70]
                    ns_range = [10]     # not used but just for consistency
                else:
                    nc_range = [30, 40, 50, 60, 70]
                    ns_range = [10, 20]
            else:
                # validate or test with best parameter setting
                if is_sample_all:
                    nc_range = [50]
                    ns_range = [10]     # not used but just for consistency
                else:
                    nc_range = [60]
                    ns_range = [20]
        num_eval = len(nc_range) * len(ns_range)

        scores_all, row_name = [], []
        if args.input_dist_level == 'clip':
            pred_id_all, vote_rate_all = [], []
            cur_vid = -1
            for ci in tqdm(range(data_len)):
                if args.debug:
                    if ci < 500:
                        continue

                data_loader, data_sampler = build_data_iterator(args, tokenizer, clip_ids=[ci])
                distances = trainer.eval_dist(data_loader)

                pred_nc_ns, voterate_nc_ns = [], []
                test_param = []
                for nc in nc_range:
                    for ns in ns_range:
                        pred, vote_rate = gen_summary(distances, clip_data, [ci], nc, ns, args.topn_exclude, is_sample_all, args.debug)
                        pred_nc_ns.append(pred[0][0])
                        voterate_nc_ns.extend(vote_rate)
                        test_param.append('nc{}_ns{}'.format(nc, ns))
                pred_id_all.append(pred_nc_ns)
                vote_rate_all.append(voterate_nc_ns)

            pred_id_all = list(zip(*pred_id_all))
            vote_rate_all = list(zip(*vote_rate_all))

        elif args.input_dist_level == 'video':
            vid_2clip = defaultdict(list)
            for ci, cdata in enumerate(clip_data):
                transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url, valid_sent_id = cdata
                vid_2clip[video_id].append(ci)
            assert len(vid_2clip) == 25     # num. of val, test videos

            pred_id_all = [[[] for _ in range(data_len)]] * num_eval
            vote_rate_all = []
            for vid, clips in (vid_2clip.items()):  # tqdm
                data_loader, data_sampler = build_data_iterator(args, tokenizer, clip_ids=clips)
                distances = trainer.eval_dist(data_loader)

                voterate_nc_ns = []
                test_param = []
                pi = 0
                for nc in nc_range:
                    for ns in ns_range:
                        pred, vote_rate = gen_summary(distances, clip_data, clips, nc, ns, args.topn_exclude, is_sample_all, args.debug)
                        for pred_id, ci in pred:
                            pred_id_all[pi][ci] = pred_id
                        voterate_nc_ns.extend(vote_rate)
                        test_param.append('nc{}_ns{}'.format(nc, ns))
                        pi += 1
                vote_rate_all.append(voterate_nc_ns)

            vote_rate_all = list(zip(*vote_rate_all))

        for pi in range(len(pred_id_all)):
            scores, params = compute_scores(args, clip_data, pred_id_all[pi], test_param[pi], save2file=False)
            scores = [score + [np.mean(vote_rate_all[pi])] for score in scores]
            # scores[0].append(np.mean(vote_rate_all[pi]))
            scores_all.extend(scores)
            row_name.extend(params)

        if args.rouge_mean:
            row_index = pd.Index(row_name, name='rows')
            df_res = pd.DataFrame(scores_all, index=row_index)
        else:
            df_res = pd.DataFrame(scores_all)
            df_res.columns = ['R1 P', 'R1 R', 'R1 F1', 'R2 P', 'R2 R', 'R2 F1', 'Rl P', 'Rl R', 'Rl F1', 'RSU4 P', 'RSU4 R', 'RSU4 F1']

        if is_sample_all:
            app_txt = ''
            app_txt_exc = '_exc-top{}'.format(args.topn_exclude) if args.topn_exclude > 0 else ''
        else:
            app_txt = '_sample'
            app_txt_exc = ''
        result_path = '{}{}_pair_c{}_e{}_{}{}{}.csv'.format(args.result_path, args.mode, args.num_cluster, args.ch_dim, args.input_dist_level, app_txt_exc, app_txt)
        df_res.to_csv(result_path)  # , header=False)


def compute_scores(args, clip_data, pred_id_all, test_param='', save2file=True):
    '''
    computes Rouge, P,R,F1, Bert-score for each generated summary
        [R-1 P, R-1 R, R-1 F1, R-2 P, R-2 R, R-2 F1, R-l P, R-l R, R-l F1, R-SU4 P, R-SU4 R, R-SU4 F1, P, P-std, R, R-std, F1, F1-std, BS-P, BS-P-std, BS-R, BS-R-std, BS-F1, BS-F1-std]
    return
        scores_all: score list of parameter settings
        test_param_all: string text of parameter settings
    '''
    # compute scores
    data_len = len(clip_data)

    scores_all = []
    test_param_all = []
    for num_sum in args.num_sum_sent:
        logger.info('num_summary_sent: {}'.format(num_sum))

        res_spread_path = '{}{}_c{}_e{}_{}_s{}.csv'.format(args.result_path, args.mode, args.num_cluster, args.ch_dim, args.input_dist_level, num_sum)
        can_path = '{}{}_c{}_e{}_{}_s{}.candidate'.format(args.result_path, args.mode, args.num_cluster, args.ch_dim, args.input_dist_level, num_sum)
        gold_path = '{}{}_c{}_e{}_{}_s{}.gold'.format(args.result_path, args.mode, args.num_cluster, args.ch_dim, args.input_dist_level, num_sum)
        save_pred = open(can_path, 'w')
        save_gold = open(gold_path, 'w')

        test_param_all.append('{}_s{}'.format(test_param, num_sum))

        p_r_f1_bin = []
        for ci in range(data_len):
            transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url, valid_sent_id = clip_data[ci]
            if isinstance(transcript[0], str):
                article = transcript
            else:
                article = [sent['display'] for sent in transcript]

            ns = num_sum if num_sum != -1 else len(e_sum)

            pred_id = pred_id_all[ci][:ns]
            pred_id.sort()
            pred = [article[pid].strip() for pid in pred_id]

            if args.ext_sum_ref:
                ext_sum_text = [sent.strip() for ei, sent in enumerate(article) if ei in e_sum]
                gold = '<q>'.join(ext_sum_text)
            else:
                abs_sum = a_sum.replace('\n', '')
                gold = '<q>'.join([sent.strip() + '.' for sent in abs_sum.split('.') if sent])

            pred = '<q>'.join(pred).replace('\n', '')

            save_pred.write(pred.strip() + '\n')
            save_gold.write(gold.strip() + '\n')

            # P, R, F1
            if e_sum is not None:
                y_true, y_pred = [], []
                for ti, sent in enumerate(article):
                    if ti in e_sum:
                        y_true.append(1)
                    else:
                        y_true.append(0)

                    if ti in pred_id:
                        y_pred.append(1)
                    else:
                        y_pred.append(0)
                # p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                # p_r_f1_w.append((p, r, f1))
                # p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
                # p_r_f1_micro.append((p, r, f1))
                # p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
                # p_r_f1_macro.append((p, r, f1))
                p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred)
                p_r_f1_bin.append((p[1], r[1], f1[1]))

        save_pred.close()
        save_gold.close()

        scores = calc_prf_rouge_bertscore(can_path, gold_path, res_spread_path, p_r_f1_bin=p_r_f1_bin, save2file=save2file, rouge_mean=args.rouge_mean)
        if not args.rouge_mean:
            scores_all.extend(scores)
        else:
            scores_all.append(scores)

    return (scores_all, test_param_all)


def gen_summary(distances, clip_loader, clip_ids, ntop_cluster, ntop_sent, topn_exclude=0, is_sample_all=True, debug=False):
    '''
    computes summary sentences given a clip list
    return
        pred_id_all: predicted (top-20) sentence id for each clip
        votes_all: voting percentage (used num. candidate sentences / all sentences) for each clip
    '''
    def popular_voting(cid, sent_st_idx, is_sample_all=True, num_sum_sent=5, debug=False):
        n_layer = 768

        transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url, valid_sent_id = clip_loader[cid]
        if isinstance(transcript[0], str):
            article = transcript
        else:
            article = [sent['display'] for sent in transcript]

        sent_end_idx = sent_st_idx + len(valid_sent_id) * n_layer

        if debug:
            print('=================================')
            print('video: {} clip:{}'.format(video_id, clip_id))

        total_vote = 0
        sent_vote = np.zeros(len(valid_sent_id))
        freq_total = sum([freq for code, freq in ca_counter])
        for code, freq in ca_counter:
            distances_ = distances[sent_st_idx:sent_end_idx]
            cluster_assign_ = cluster_assign[sent_st_idx:sent_end_idx]
            cand_idx = [i for i, c in enumerate(cluster_assign_) if c == code]
            dist_c = distances_[cand_idx, code]

            dist_c_id = [(d, cand_idx[i]) for i, d in enumerate(dist_c)]
            dist_c_id.sort(key=lambda x: x[0])

            # count total num. of vote
            if is_sample_all:
                total_vote += len(dist_c_id)
                dist_cand = dist_c_id
            else:
                total_vote += len(dist_c_id[:ntop_sent])
                dist_cand = dist_c_id[:ntop_sent]

            for i, (dist, di) in enumerate(dist_cand):
                votes_id = di // n_layer
                if True:
                    # hard voting
                    sent_vote[votes_id] += 1
                    # weigthed hard voting
                    # sent_vote[votes_id] += (freq / freq_total)

        sent_vote_id = [(i, v) for i, v in enumerate(sent_vote)]
        sent_vote_id.sort(key=lambda x: x[1], reverse=True)
        if debug and 0:
            sent_vote_id_ = [(valid_sent_id[i], v) for i, v in sent_vote_id]
            print ('=> top-{0} (at most {0}) utterance voting from top-{1} codes (sent ID, votes) - sorted by votes\n'.format(ntop_sent, ntop_cluster),
                   len(sent_vote_id_), sent_vote_id_, sum([v for i, v in sent_vote_id_]), '\n')

        if debug:
            sent_assign_code = []
            for code, freq in ca_counter_debug:
                distances_ = distances[sent_st_idx:sent_end_idx]
                cluster_assign_ = cluster_assign[sent_st_idx:sent_end_idx]
                cand_idx = [i for i, c in enumerate(cluster_assign_) if c == code]
                dist_c = distances_[cand_idx, code]

                dist_c_id = [(d, cand_idx[i]) for i, d in enumerate(dist_c)]
                dist_c_id.sort(key=lambda x: x[0])

                # for debug
                sent_vote_code = np.zeros(len(valid_sent_id))
                for d, idx in dist_c_id:
                    vid = idx // n_layer
                    sent_vote_code[vid] += 1
                sent_vote_sorted = sorted([(valid_sent_id[i], vote) for i, vote in enumerate(sent_vote_code)], key=lambda x: x[1], reverse=True)
                sent_assign_code.append(sent_vote_sorted)

            # print ('total num. of vote: {}/{}'.format(total_vote, len(valid_sent_id) * n_layer))
            # print ('=> top-20 (at most 20) utterances (and assigned freq.) assigned to each code - sorted by freq.')
            # for sai, sent_vote_code_ in enumerate(sent_assign_code):
            #     print ('code:{} (sent ID, assigned freq.)\n'.format(ca_counter[sai]), sent_vote_code_[:20])
            # print ('\n', end='')
            print ('=> top-5 utterances (and assigned freq.) assigned to top-{} code - sorted by freq.'.format(len(ca_counter_debug)))
            for sai, sent_vote_code_ in enumerate(sent_assign_code):
                print ('code:{}'.format(ca_counter_debug[sai]))
                for svi in range(5):
                    sent_id, vote = sent_vote_code_[svi]
                    print ('[{}] {} (vote:{})'.format(sent_id, article[sent_id], vote))
                print ('\n')
            print ('\n', end='')

        def debug_print(sent_cand_id):
            print ('=> [utterance ID * &] utterance')
            print ('=> * : summary by annotator, & : summary by system, -ignore-: sentence less than 5 words')
            for si, sent in enumerate(article):
                mark_ext = '*' if si in e_sum else ''
                mark_cand_sent = '&' if si in sent_cand_id else ''
                if si in valid_sent_id:
                    print('[{:02} {} {}] {}'.format(si, mark_ext, mark_cand_sent, sent))
                else:
                    print('[{:02} -ignored-] {}'.format(si, sent))

            print ('\n=> extractive summary, abstractive summary')
            print(e_sum, a_sum)
            print('=================================')

        n_sys_sent = len(e_sum) if num_sum_sent == -1 else num_sum_sent

        # for debug
        system_summary_id = [valid_sent_id[i] for i, v in sent_vote_id[:n_sys_sent]]
        if debug:
            debug_print(system_summary_id)

        # return summary - enough number (20) of sentences fo one compuatation
        # system_summary_id = [valid_sent_id[i] for i, v in sent_vote_id[:20]]
        # pred = [sent for si, sent in enumerate(article) if si in set(system_summary_id)]

        # enough prediction selection until 20th
        # pred = [article[valid_sent_id[i]] for i, v in sent_vote_id[:20]]
        pred_id = [valid_sent_id[i] for i, v in sent_vote_id[:20]]

        return (pred_id, sent_end_idx, total_vote / (len(valid_sent_id) * n_layer))

    # top-P clusters
    cluster_assign = np.argmin(distances, axis=1)

    ca_counter = Counter(cluster_assign)
    ca_counter = ca_counter.most_common(ntop_cluster)
    if is_sample_all:
        ca_counter_debug = ca_counter[:3]
        ca_counter = ca_counter[topn_exclude:]
    if debug:
        print ('=> top-{} codes (code ID, votes)\n'.format(ntop_cluster), len(ca_counter), ca_counter, '\n')

    pred_id_all = []
    votes_all = []
    clip_st_idx = 0
    for cid in clip_ids:
        prd_id, clip_st_idx, vote_rate = popular_voting(cid, clip_st_idx, is_sample_all=is_sample_all, debug=debug)
        pred_id_all.append((prd_id, cid))
        votes_all.append(vote_rate)
        if debug:
            pdb.set_trace()

    return (pred_id_all, votes_all)


class VQVAEOptimizer(object):
    # not used
    def __init__(self, model, lr, warmup_steps, beta_1=0.99, beta_2=0.999, eps=1e-8):
        self.encoder_param = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert.model')]
        self.not_encoder_param = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert.model')]

        self.lr = lr
        self.warmup_steps = warmup_steps

        self.optimizers = {
            "encoder": torch.optim.Adam(
                self.encoder_param, lr=lr["encoder"], betas=(beta_1, beta_2), eps=eps,
            ),
            "not_encoder": torch.optim.Adam(
                self.not_encoder_param, lr=lr["not_encoder"], betas=(beta_1, beta_2), eps=eps,
            ),
        }

        self._step = 0
        self.current_learning_rates = {}

    def _update_rate(self, stack):
        return self.lr[stack] * min(self._step ** (-0.5), self._step * self.warmup_steps[stack] ** (-1.5))

    def zero_grad(self):
        self.optimizer_decoder.zero_grad()
        self.optimizer_encoder.zero_grad()

    def step(self):
        self._step += 1
        for stack, optimizer in self.optimizers.items():
            new_rate = self._update_rate(stack)
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_rate
            optimizer.step()
            self.current_learning_rates[stack] = new_rate


if __name__ == '__main__':
    main()
