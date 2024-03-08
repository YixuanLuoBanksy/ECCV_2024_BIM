# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

'''
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
'''

def train_one_epoch(model_block1: torch.nn.Module,
                    model_block2: torch.nn.Module,
                    model_block3: torch.nn.Module,
                    model_block4: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer_block1: torch.optim.Optimizer,
                    optimizer_block2: torch.optim.Optimizer,
                    optimizer_block3: torch.optim.Optimizer,
                    optimizer_block4: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler_block1,
                    loss_scaler_block2,
                    loss_scaler_block3,
                    loss_scaler_block4,
                    cos_similarity_store,
                    avg_masking_ratio_record,
                    log_writer=None,
                    args=None):
    model_block1.train(True)
    model_block2.train(True)
    model_block3.train(True)
    model_block4.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer_block1.zero_grad()
    optimizer_block2.zero_grad()
    optimizer_block3.zero_grad()
    optimizer_block4.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer_block1, data_iter_step / len(data_loader) + epoch, args)
            lr_sched.adjust_learning_rate(optimizer_block2, data_iter_step / len(data_loader) + epoch, args)
            lr_sched.adjust_learning_rate(optimizer_block3, data_iter_step / len(data_loader) + epoch, args)
            lr_sched.adjust_learning_rate(optimizer_block4, data_iter_step / len(data_loader) + epoch, args)
        samples = samples.to(device, non_blocking=True)

        # block 1
        with torch.cuda.amp.autocast():
            # loss, mask, patches, ids_restore = model_block1(samples, mask_ratio=args.mask_ratio)
            loss, mask, patches, ids_restore, avg_cos_similarity = model_block1(samples, cos_similarity_store[0])
            cos_similarity_store[0] = avg_cos_similarity[0]
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler_block1(loss, optimizer_block1, parameters=model_block1.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_block1.zero_grad()
        torch.cuda.synchronize()

        # block 2
        with torch.cuda.amp.autocast():
            #loss, mask, patches, ids_restore = model_block2(img, patches.detach(), mask, ids_restore)
            loss, mask, patches, ids_restore, avg_cos_similarity = model_block2(samples, patches, mask, ids_restore, cos_similarity_store[0], cos_similarity_store[1])
            cos_similarity_store[1] = avg_cos_similarity.mean()
        loss_value += loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler_block2(loss, optimizer_block2, parameters=model_block2.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_block2.zero_grad()
        torch.cuda.synchronize()  

        # block 3
        with torch.cuda.amp.autocast():
            #loss, mask, patches, ids_restore = model_block3(img, patches.detach(), mask, ids_restore)
            loss, mask, patches, ids_restore, avg_cos_similarity = model_block2(samples, patches, mask, ids_restore, cos_similarity_store[1], cos_similarity_store[2])
            cos_similarity_store[2] = avg_cos_similarity.mean()
        loss_value += loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler_block3(loss, optimizer_block3, parameters=model_block3.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_block3.zero_grad()
        torch.cuda.synchronize()  

        # block 4
        with torch.cuda.amp.autocast():
            loss, mask, patches, ids_restore, avg_cos_similarity = model_block2(samples, patches, mask, ids_restore, cos_similarity_store[2], cos_similarity_store[3])
            cos_similarity_store[3] = avg_cos_similarity.mean()
        loss_value += loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler_block4(loss, optimizer_block4, parameters=model_block4.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer_block4.zero_grad()
        torch.cuda.synchronize()  

        metric_logger.update(loss=loss_value)

        lr = optimizer_block1.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        avg_masking_ratio_record_temp = []
        for i in range(len(cos_similarity_store)):
            cos_similarity_store[i] = misc.all_reduce_mean(cos_similarity_store[i])
            avg_masking_ratio_record_temp.append(cos_similarity_store[i])
        avg_masking_ratio_record.append(avg_masking_ratio_record_temp)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, cos_similarity_store