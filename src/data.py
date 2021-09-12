import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

Batch = namedtuple("Batch", ["batch_size", "src", "seg", "mask_src", "tgt", "src_text"])


class BehanceDataset(Dataset):
    def __init__(self, datapath="", split="train", num_min_word=1, debug=False, clip_ids=[]):
        '''
        datapath: path to Behance data
        split: train, val, test
        num_min_word: num. of min word to be selected as a candidate sentence
        debug: debug mode
        clip_ids: list of clip ids for val, test
        '''
        assert os.path.isdir(datapath)
        split_ori = split
        split = 'train' if 'train' in split else split
        split = 'test' if 'test' in split else split
        split = 'val' if 'val' in split else split
        with open(os.path.join(datapath, 'Behance_{}.pkl'.format(split)), 'rb') as fp:
            # data: [[transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url], ...]
            data = pickle.load(fp)

        self.sentences = []
        self.num_skipped_item = 0
        if split_ori == 'train':
            if debug:
                data = data[:10]

            for clip in data:
                transcript = clip[0]
                for sentence in transcript:
                    sent = sentence['display']
                    if len(sent.split()) >= num_min_word:
                        self.sentences.append(sent.lower().strip())
                    else:
                        self.num_skipped_item += 1
        else:
            for cid in clip_ids:
                clip = data[cid]
                transcript = clip[0]
                for sentence in transcript:
                    sent = sentence['display']
                    if len(sent.split()) >= num_min_word:
                        self.sentences.append(sent.lower().strip())
                    else:
                        self.num_skipped_item += 1

    def get_skip_trans_cnt(self):
        return self.num_skipped_item

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class BehanceClipDataset(Dataset):
    def __init__(self, datapath="", split="test", num_min_word=1, debug=False):
        assert os.path.isdir(datapath)
        split = 'train' if 'train' in split else split
        split = 'test' if 'test' in split else split
        split = 'val' if 'val' in split else split
        with open(os.path.join(datapath, 'Behance_{}.pkl'.format(split)), 'rb') as fp:
            self.clip_data = pickle.load(fp)

        if debug:
            self.clip_data = self.clip_data[:10]

        new_clip_data = []
        for clip in self.clip_data:
            transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url = clip
            valid_sent_id = [i for i, sent in enumerate(transcript) if len(sent['display'].split()) >= num_min_word]

            new_clip = (transcript, a_sum, e_sum, video_id, clip_id, vid_title, vid_url, transcript_url, valid_sent_id)
            new_clip_data.append(new_clip)
        self.clip_data = new_clip_data

    def __len__(self):
        return len(self.clip_data)

    def __getitem__(self, idx):
        return self.clip_data[idx]


def build_data_iterator(args, tokenizer, clip_ids=[]):

    dataset = BehanceDataset(args.data_path, args.mode, args.num_min_word, debug=args.debug, clip_ids=clip_ids)
    # print('num data: {}'.format(len(dataset)))
    # print('num_skipped_item', dataset.get_skip_trans_cnt())

    # data_sampler = DistributedSampler(dataset) if args.distributed else None
    data_sampler = None

    def collate_wrapper(batch):
        return collateBatch(batch, tokenizer, block_size=args.max_pos)

    data_iterator = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(args.mode == 'train'),  # (data_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=data_sampler, collate_fn=collate_wrapper)

    return data_iterator, data_sampler


class collateBatch:
    def __init__(self, batch_data, tokenizer, block_size):
        symbols = {'BOS': tokenizer.vocab['[unused0]'],
                   'EOS': tokenizer.vocab['[unused1]'],
                   'PAD': tokenizer.vocab['[PAD]'],
                   'EOQ': tokenizer.vocab['[unused2]'],
                   'SEP': tokenizer.vocab['[SEP]'],
                   'CLS': tokenizer.vocab['[CLS]'], }

        encoded_transcript = [tokenizer.encode(sent, add_special_tokens=False) for sent in batch_data]
        encoder_transcript = [[symbols['CLS']] + sent + [symbols['SEP']] for sent in encoded_transcript]
        decoder_transcript = [[symbols['BOS']] + sent + [symbols['EOS']] for sent in encoded_transcript]

        encoder_transcript_tensor = torch.tensor([truncate_or_pad(sent, block_size, tokenizer.pad_token_id) for sent in encoder_transcript])
        encoder_mask = build_mask(encoder_transcript_tensor, tokenizer.pad_token_id)
        encoder_seg = torch.zeros_like(encoder_mask)

        decoder_transcript_tensor = torch.tensor([truncate_or_pad(sent, block_size, tokenizer.pad_token_id) for sent in decoder_transcript])
        # decoder_mask = build_mask(decoder_transcript_tensor, tokenizer.pad_token_id)      # not used

        self.batch = Batch(
            batch_size=len(encoder_transcript_tensor),
            src=encoder_transcript_tensor,
            seg=encoder_seg,
            mask_src=encoder_mask,
            tgt=decoder_transcript_tensor,
            src_text=batch_data,
        )


def truncate_or_pad(sequence, block_size, pad_token_id):
    """ Adapt the source and target sequences' lengths to the block size.
    If the sequence is shorter we append padding token to the right of the sequence.
    """
    if len(sequence) > block_size:
        return sequence[:block_size]
    else:
        sequence.extend([pad_token_id] * (block_size - len(sequence)))
        return sequence


def build_mask(sequence, pad_token_id):
    """ Builds the mask. The attention mechanism will only attend to positions
    with value 1. """
    mask = torch.ones_like(sequence)
    idx_pad_tokens = sequence == pad_token_id
    mask[idx_pad_tokens] = 0
    return mask
