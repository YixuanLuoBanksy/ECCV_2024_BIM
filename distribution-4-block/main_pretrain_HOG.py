# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_HOG, models_block_wise_HOG

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # mask ratio bound parameters
    parser.add_argument('--fixed_masking', action='store_true',
                        help='Use fixed mask ratio')
    parser.set_defaults(fixed_masking=True)
    parser.add_argument('--dynamic_lower_bound', default=0.75, type=float,
                        help='number of distributed processes')
    parser.add_argument('--dynamic_upper_bound', default=0.85, type=float,
                        help='number of distributed processes')
    
    parser.add_argument('--mask_ratio_layer1', default=0.65, type=float,
                        help='number of distributed processes')
    parser.add_argument('--mask_ratio_layer2', default=0.70, type=float,
                        help='number of distributed processes')
    parser.add_argument('--mask_ratio_layer3', default=0.80, type=float,
                        help='number of distributed processes')
    parser.add_argument('--mask_ratio_layer4', default=0.85, type=float,
                        help='number of distributed processes')

    # decoder depth
    parser.add_argument('--decoder_depth_layer1', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--decoder_depth_layer2', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--decoder_depth_layer3', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--decoder_depth_layer4', default=8, type=int,
                        help='number of distributed processes')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    if args.model == 'mae_vit_base_patch16':
        model_block1 = Block1(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4, 
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer1)
        model_block2 = Block2(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer2)
        model_block3 = Block2(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer3)
        model_block4 = Block2(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4, 
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer4)
    elif args.model == 'mae_vit_large_patch16':
        model_block1 = Block1(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer1)
        model_block2 = Block2(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer2)
        model_block3 = Block2(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer3)
        model_block4 = Block2(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer4)
    elif args.model == 'mae_vit_huge_patch14':
        model_block1 = Block1(
            patch_size=14, embed_dim=1280, depth=32, num_heads=16,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer1)
        model_block2 = Block2(
            patch_size=14, embed_dim=1280, depth=32, num_heads=16,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer2)
        model_block3 = Block2(
            patch_size=14, embed_dim=1280, depth=32, num_heads=16,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer3)
        model_block4 = Block2(
            patch_size=14, embed_dim=1280, depth=32, num_heads=16,
            decoder_embed_dim=512, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), block_num=4,
            norm_pix_loss=args.norm_pix_loss, fixed_masking=args.fixed_masking, 
            fixed_mask_ratio=args.fixed_mask_ratio,dynamic_lower_bound=args.dynamic_lower_bound, 
            dynamic_upper_bound=args.dynamic_upper_bound, decoder_depth=args.decoder_depth_layer4)
    
    else:
        raise Exception("Model Support Error: no implementation for", args.model)

    '''
    models = model_block_wise.__dict__[args.model](norm_pix_loss=args.norm_pix_loss, mask_ratio_low_bound=args.mask_ratio_lower_bound, mask_ratio_up_bound=args.mask_ratio_upper_bound)
    model_block1 = models[0]
    model_block2 = models[1]
    model_block3 = models[2]
    model_block4 = models[3]
    '''
    # model.to(device)
    model_block1.to(device)
    model_block2.to(device)
    model_block3.to(device)
    model_block4.to(device)

    # model_without_ddp = model
    # print("Model = %s" % str(model_without_ddp))
    model_without_ddp_block1 = model_block1
    model_without_ddp_block2 = model_block2
    model_without_ddp_block3 = model_block3
    model_without_ddp_block4 = model_block4    
    print("Model Block1 = %s" % str(model_without_ddp_block1))
    print("Model Block2 = %s" % str(model_without_ddp_block2))
    print("Model Block3 = %s" % str(model_without_ddp_block3))
    print("Model Block4 = %s" % str(model_without_ddp_block4))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # model_without_ddp = model.module
        model_block1 = torch.nn.parallel.DistributedDataParallel(model_block1, device_ids=[args.gpu], find_unused_parameters=True)
        model_block2 = torch.nn.parallel.DistributedDataParallel(model_block2, device_ids=[args.gpu], find_unused_parameters=True)
        model_block3 = torch.nn.parallel.DistributedDataParallel(model_block3, device_ids=[args.gpu], find_unused_parameters=True)
        model_block4 = torch.nn.parallel.DistributedDataParallel(model_block4, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp_block1 = model_block1.module
        model_without_ddp_block2 = model_block2.module
        model_without_ddp_block3 = model_block3.module
        model_without_ddp_block4 = model_block4.module

    # update until here
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    # loss_scaler = NativeScaler()
    param_groups_block1 = optim_factory.add_weight_decay(model_without_ddp_block1, args.weight_decay)
    param_groups_block2 = optim_factory.add_weight_decay(model_without_ddp_block2, args.weight_decay)
    param_groups_block3 = optim_factory.add_weight_decay(model_without_ddp_block3, args.weight_decay)
    param_groups_block4 = optim_factory.add_weight_decay(model_without_ddp_block4, args.weight_decay)
    optimizer_block1 = torch.optim.AdamW(param_groups_block1, lr=args.lr, betas=(0.9, 0.95))
    optimizer_block2 = torch.optim.AdamW(param_groups_block2, lr=args.lr, betas=(0.9, 0.95))
    optimizer_block3 = torch.optim.AdamW(param_groups_block3, lr=args.lr, betas=(0.9, 0.95))
    optimizer_block4 = torch.optim.AdamW(param_groups_block4, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler_block1 = NativeScaler()
    loss_scaler_block2 = NativeScaler()
    loss_scaler_block3 = NativeScaler()
    loss_scaler_block4 = NativeScaler()
    
    cos_similarity_store = [0,0,0,0]

    misc.load_model(args=args, model_without_ddp=model_without_ddp_block1, optimizer=optimizer_block1, loss_scaler=loss_scaler_block1, cos_similarity_store=cos_similarity_store, block_index=1)
    misc.load_model(args=args, model_without_ddp=model_without_ddp_block2, optimizer=optimizer_block2, loss_scaler=loss_scaler_block2, cos_similarity_store=cos_similarity_store, block_index=2)
    misc.load_model(args=args, model_without_ddp=model_without_ddp_block3, optimizer=optimizer_block3, loss_scaler=loss_scaler_block3, cos_similarity_store=cos_similarity_store, block_index=3)
    misc.load_model(args=args, model_without_ddp=model_without_ddp_block4, optimizer=optimizer_block4, loss_scaler=loss_scaler_block4, cos_similarity_store=cos_similarity_store, block_index=4)

    avg_masking_ratio_record = []


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats, cos_similarity_store = train_one_epoch(model_without_ddp_block1,
                    model_without_ddp_block2,
                    model_without_ddp_block3,
                    model_without_ddp_block4,
                    data_loader_train,
                    optimizer_block1,
                    optimizer_block2,
                    optimizer_block3,
                    optimizer_block4,
                    device, 
                    epoch, 
                    loss_scaler_block1,
                    loss_scaler_block2,
                    loss_scaler_block3,
                    loss_scaler_block4,
                    avg_masking_ratio_record,
                    log_writer=log_writer,
                    cos_similarity_store = cos_similarity_store,
                    args=args)
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model_block1, model_without_ddp=model_without_ddp_block1, optimizer=optimizer_block1,
                loss_scaler=loss_scaler_block1, epoch=epoch, cos_similarity_store=cos_similarity_store, block_index=1)
            misc.save_model(
                args=args, model=model_block2, model_without_ddp=model_without_ddp_block2, optimizer=optimizer_block2,
                loss_scaler=loss_scaler_block2, epoch=epoch, cos_similarity_store=cos_similarity_store,block_index=2)
            misc.save_model(
                args=args, model=model_block3, model_without_ddp=model_without_ddp_block3, optimizer=optimizer_block3,
                loss_scaler=loss_scaler_block3, epoch=epoch, cos_similarity_store=cos_similarity_store,block_index=3)
            misc.save_model(
                args=args, model=model_block4, model_without_ddp=model_without_ddp_block4, optimizer=optimizer_block4,
                loss_scaler=loss_scaler_block4, epoch=epoch, cos_similarity_store=cos_similarity_store,block_index=4)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir and misc.is_main_process():
            df = pd.DataFrame(avg_masking_ratio_record, columns =['block-1', 'block-2', 'block-3', 'block-4'])
            df.to_csv(os.path.join(args.output_dir, "masking.csv")) 

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
