import os

import distributed
import numpy as np
import torch

from models.reporter import ReportMgr, Statistics
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.logging import logger


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    return n_params


def build_trainer(args, model, optims, loss, num_steps=0):

    grad_accum_count = args.accum_count
    n_gpu = args.n_gpu
    gpu_rank = 0 if args.gpu is None else args.gpu

    logger.info('gpu_rank %d' % gpu_rank)

    writer = SummaryWriter(args.model_path, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optims, loss, num_steps, grad_accum_count, n_gpu, gpu_rank, report_manager)

    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of trainable parameters: %d' % n_params)

    return trainer


class Trainer(object):

    def __init__(self, args, model, optims, loss, num_steps=0,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optims = optims
        self.num_steps = num_steps
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = loss

        assert grad_accum_count > 0
        # Set model in training mode.
        if (model):
            self.model.train()

    def train(self, train_loader, epoch, valid_iter_fct=None, valid_steps=-1):

        logger.info('Start training epoch {}...'.format(epoch))
        step = self.optims[0]._step + 1

        true_batchs = []
        accum = 0
        normalization = 0

        num_data_epoch = len(train_loader.dataset)
        batch_size_sum = 0

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        with tqdm(total=num_data_epoch) as tbar:
            for i, batch in enumerate(train_loader):

                batch_size = batch.batch.batch_size
                true_batchs.append(batch)
                num_tokens = batch.batch.tgt[:, 1:].ne(self.loss.padding_idx).sum()
                normalization += num_tokens.item()
                accum += 1
                if accum == self.grad_accum_count:

                    if self.n_gpu > 1:
                        normalization = sum(distributed.all_gather_list(normalization))

                    self._gradient_accumulation(true_batchs, normalization, total_stats,
                                                report_stats, epoch)

                    learning_rate = [self.optims[0].learning_rate, self.optims[1].learning_rate] if len(self.optims) == 2 else self.optims[0].learning_rate
                    report_stats = self._maybe_report_training(step, self.num_steps,
                                                               learning_rate,
                                                               report_stats)

                    batch_size_sum = batch_size * accum
                    tbar.update(batch_size_sum)

                    true_batchs = []
                    accum = 0
                    normalization = 0

                    if step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0:
                        self._save(step, epoch)
                    step += 1

        if epoch == self.args.epochs - 1 and self.gpu_rank == 0:
            self._save(step, epoch)

        return total_stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats, epoch):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            # batch_size = batch.batch_size
            src = batch.batch.src.cuda(self.gpu_rank, non_blocking=True)
            seg = batch.batch.seg.cuda(self.gpu_rank, non_blocking=True)
            mask_src = batch.batch.mask_src.cuda(self.gpu_rank, non_blocking=True)
            tgt = batch.batch.tgt.cuda(self.gpu_rank, non_blocking=True)

            outputs, vq_outputs = self.model(src, tgt, seg, mask_src)
            if vq_outputs is not None:
                loss_vq = vq_outputs[0]
            else:
                loss_vq = None

            batch_stats = self.loss.compute_loss_all(batch.batch, outputs, normalization, loss_vq, self.gpu_rank)

            batch_stats.n_docs = int(src.size(0))

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # update the parameters and statistics
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))

                for o in self.optims:
                    o.step()

        # in case of multi step gradient accumulation, update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            for o in self.optims:
                o.step()

    def eval_dist(self, test_loader):
        self.model.eval()

        distances_list = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):

                src = batch.batch.src.cuda(self.gpu_rank, non_blocking=True)
                seg = batch.batch.seg.cuda(self.gpu_rank, non_blocking=True)
                mask_src = batch.batch.mask_src.cuda(self.gpu_rank, non_blocking=True)
                tgt = batch.batch.tgt.cuda(self.gpu_rank, non_blocking=True)
                text = batch.batch.src_text

                outputs, vq_outputs = self.model(src, tgt, seg, mask_src)
                loss, quantized, perplexity, encodings, distances = vq_outputs
                distances_list.append(distances.data.cpu().numpy())     # (B, E)

        distances = np.concatenate(distances_list, axis=0)
        return distances

    def _save(self, step, epoch):

        checkpoint = {
            'epoch': epoch + 1,
            'model': self.model.state_dict(),
            'opt': self.args,
            'optims': self.optims,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_{}_epoch_{}.pt'.format(step, epoch))
        logger.info("Saving checkpoint %s" % checkpoint_path)

        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
