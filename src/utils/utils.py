import os
import re
import shutil
import errno
import json
import time
import numpy as np
import pandas as pd

from bert_score import BERTScorer

from utils import pyrouge

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def save_json(data_json, file):
    with open(file, 'w') as fj:
        json.dump(data_json, fj)


def print_stats_int(data, desc=''):
    print('{} - len:{}, min:{}, max:{}, mean:{:.3f}, median:{:.3f}, std:{:.3f}, sum:{}'.format(desc, len(data), np.min(data), np.max(data), np.mean(data), np.median(data), np.std(data), np.sum(data)))


def calc_prf_rouge_bertscore(can_path, gold_path, result_path=None, is_bertscore=False, p_r_f1_bin=None, save2file=True, rouge_mean=True):
    if save2file and result_path is None:
        print('no result file path given...')
        return

    csv_results = []

    # Rouge output
    rouge_outputs, avg_words, avg_sents = test_rouge('../temp', can_path, gold_path, rouge_mean=rouge_mean)
    if not rouge_mean:
        if save2file:
            df_res = pd.DataFrame(rouge_outputs)
            df_res.columns = ['R1 P', 'R1 R', 'R1 F1', 'R2 P', 'R2 R', 'R2 F1', 'Rl P', 'Rl R', 'Rl F1', 'RSU4 P', 'RSU4 R', 'RSU4 F1']
            df_res.to_csv(result_path)
        return rouge_outputs

    csv_results.extend(rouge_outputs)

    # P, R, F1 output
    if p_r_f1_bin is not None:
        # outputs = print_prf(p_r_f1_w, 'weighted')
        # outputs = print_prf(p_r_f1_micro, 'micro')
        # outputs = print_prf(p_r_f1_macro, 'macro')
        outputs = print_prf(p_r_f1_bin, 'label-1')
        csv_results.extend(outputs)
    else:
        csv_results.extend(['', '', '', '', '', ''])

    # BERT score P, R, F1 output
    if is_bertscore:
        scorer = BERTScorer(lang="en", rescale_with_baseline=False)
        with open(can_path) as f:
            cands = [line.strip() for line in f]
        with open(gold_path) as f:
            refs = [line.strip() for line in f]
        assert len(cands) == len(refs)

        bert_p_r_f1 = []
        for ci, cand in enumerate(cands):
            ref = refs[ci]
            # input must be a list
            if cand.find('<q>') >= 0:
                cand_sents = cand.split('<q>')
            else:
                # abstractive summary
                cand_sents = cand.split('.')
            ref_sents = ref.split('<q>')
            P, R, F1 = scorer.score(cand_sents, [ref_sents])
            bert_p_r_f1.append((P, R, F1))

        outputs = print_prf(bert_p_r_f1, 'BERT_score', scale100=False)
        csv_results.extend(outputs)

    # num. words in system summary
    csv_results.extend([avg_words, avg_sents])

    if save2file:
        df_res = pd.DataFrame(csv_results)
        df_res = df_res.T
        df_res.to_csv(result_path, index=False, header=False)

    return csv_results


def process(params):
    temp_dir, data = params
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}-{}".format(current_time, pool_id))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def remove_split_char(str_list, split_ch='<q>'):
    new_str_list = []
    num_sents_pc, num_words_pc, num_words_ps = [], [], []
    for str_line in str_list:
        # if str_line.find(split_ch) >= 0:
        sents = str_line.split(split_ch)
        num_sents_pc.append(len(sents))
        nw_ps = [len(utt.split()) for utt in sents]
        num_words_ps.extend(nw_ps)
        num_words_pc.append(sum(nw_ps))
        str_line = ' '.join(sents)
        new_str_list.append(str_line)
    return (new_str_list, num_sents_pc, num_words_pc, num_words_ps)


def test_rouge(temp_dir, cand, ref, split_ch='<q>', rouge_mean=True):
    candidates = [line.strip() for line in open(cand, encoding='utf-8')]
    references = [line.strip() for line in open(ref, encoding='utf-8')]
    candidates, num_sents_pc, num_words_pc, num_words_ps = remove_split_char(candidates, split_ch)
    references, num_sents_pc_ref, num_words_pc_ref, num_words_ps_ref = remove_split_char(references, split_ch)
    print('candidates:', len(candidates))
    print('references:', len(references))
    assert len(candidates) == len(references), '{} != {}'.format(len(candidates), len(references))
    print_stats_int(num_sents_pc, 'num_sents_pc')
    print_stats_int(num_words_pc, 'num_words_pc')
    print_stats_int(num_words_ps, 'num_words_ps')
    print_stats_int(num_sents_pc_ref, 'num_sents_pc_ref')
    print_stats_int(num_words_pc_ref, 'num_words_pc_ref')
    print_stats_int(num_words_ps_ref, 'num_words_ps_ref')

    # avg_words = np.mean([len(summary.split()) for summary in candidates])
    avg_words = np.mean(num_words_pc)
    avg_sents = np.mean(num_sents_pc)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        if rouge_mean:
            for i in range(cnt):
                if len(references[i]) < 1:
                    continue
                with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                          encoding="utf-8") as f:
                    f.write(candidates[i])
                with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                          encoding="utf-8") as f:
                    f.write(references[i])
            r = pyrouge.Rouge155(temp_dir=temp_dir)
            r.model_dir = tmp_dir + "/reference/"
            r.system_dir = tmp_dir + "/candidate/"
            r.model_filename_pattern = 'ref.#ID#.txt'
            r.system_filename_pattern = r'cand.(\d+).txt'
            rouge_results = r.convert_and_evaluate()
            # print(rouge_results)
            results_dict = r.output_to_dict(rouge_results)
            rouge_outputs = print_rouge_scores(results_dict)

        else:
            # reouge for each test sample
            rouge_outputs = []
            for i in range(cnt):
                if len(references[i]) < 1:
                    continue
                cand_path = tmp_dir + "/candidate/cand.{}.txt".format(i)
                with open(cand_path, "w", encoding="utf-8") as f:
                    f.write(candidates[i])
                ref_path = tmp_dir + "/reference/ref.{}.txt".format(i)
                with open(ref_path, "w", encoding="utf-8") as f:
                    f.write(references[i])
                r = pyrouge.Rouge155(temp_dir=temp_dir)
                r.model_dir = tmp_dir + "/reference/"
                r.system_dir = tmp_dir + "/candidate/"
                r.model_filename_pattern = 'ref.#ID#.txt'
                r.system_filename_pattern = r'cand.(\d+).txt'
                rouge_results = r.convert_and_evaluate()
                # print(rouge_results)
                results_dict = r.output_to_dict(rouge_results)
                rouges = print_rouge_scores(results_dict)
                rouge_outputs.append(rouges)

                if os.path.exists(cand_path):
                    os.remove(cand_path)
                if os.path.exists(ref_path):
                    os.remove(ref_path)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)

    return (rouge_outputs, avg_words, avg_sents)


def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_recall"] * 100

        # ,results_dict["rouge_su*_f_score"] * 100
    )


def print_rouge_scores(rouges):
    r1p, r1r, r1f = rouges['rouge_1_precision'] * 100, rouges['rouge_1_recall'] * 100, rouges['rouge_1_f_score'] * 100
    r2p, r2r, r2f = rouges['rouge_2_precision'] * 100, rouges['rouge_2_recall'] * 100, rouges['rouge_2_f_score'] * 100
    rlp, rlr, rlf = rouges['rouge_l_precision'] * 100, rouges['rouge_l_recall'] * 100, rouges['rouge_l_f_score'] * 100
    rs4p, rs4r, rs4f = rouges['rouge_su4_precision'] * 100, rouges['rouge_su4_recall'] * 100, rouges['rouge_su4_f_score'] * 100
    outputs = [r1p, r1r, r1f, r2p, r2r, r2f, rlp, rlr, rlf, rs4p, rs4r, rs4f]
    print('Rouge-1: {:.2f} {:.2f} {:.2f}\nRouge-2: {:.2f} {:.2f} {:.2f}\nRouge-l: {:.2f} {:.2f} {:.2f}\nRouge-SU4: {:.2f} {:.2f} {:.2f}\n'.format(*outputs))
    outputs = [float('{:.2f}'.format(out)) for out in outputs]
    return outputs


def print_prf(p_r_f1, prefix='', scale100=True):
    precision, recall, f1_score = [], [], []
    for i, (p, r, f1) in enumerate(p_r_f1):
        precision.append(p)
        recall.append(r)
        f1_score.append(f1)
    outputs = [np.mean(precision), np.std(precision), np.mean(recall), np.std(recall), np.mean(f1_score), np.std(f1_score)]
    if scale100:
        outputs = [elem * 100 for elem in outputs]
        print ('[{}] Avg. Precision:{:.2f} (std:{:.2f}), Avg. Recall:{:.2f} (std:{:.2f}), Avg. F1:{:.2f} (std:{:.2f})'.format(prefix, *outputs))
        outputs = [float('{:.2f}'.format(out)) for out in outputs]
    else:
        print ('[{}] Avg. Precision:{:.3f} (std:{:.3f}), Avg. Recall:{:.3f} (std:{:.3f}), Avg. F1:{:.3f} (std:{:.3f})'.format(prefix, *outputs))
        outputs = [float('{:.3f}'.format(out)) for out in outputs]
    return outputs
