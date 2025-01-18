import os
import time
import random
import numpy as np
import logging
import argparse
import shutil
import zlib
import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import (
    collate_fn_limit,
    collation_fn_voxelmean,
    collation_fn_voxelmean_tta,
)
from util.logger import get_logger
from util.lr import MultiStepWithWarmup, PolyLR, PolyLRwithWarmup, Constant

from dataset.boreas_nuscenes import boreas_nuScenes
from functools import partial
import pickle
import yaml
from torch_scatter import scatter_mean
import spconv.pytorch as spconv

def get_file_names(dataroot):
    lidar_names = [
        f for f in os.listdir(dataroot) if os.path.isfile(os.path.join(dataroot, f))
    ]
    lidar_file_names_without_extension = [os.path.splitext(f)[0] for f in lidar_names]
    lidar_file_names = np.array(lidar_file_names_without_extension, dtype=np.int64)
    lidar_file_names = np.sort(lidar_file_names)
    return lidar_file_names


def get_parser():
    parser = argparse.ArgumentParser(
        description="PyTorch Point Cloud Semantic Segmentation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/s3dis/s3dis_stratified_transformer.yaml",
        help="config file",
    )
    parser.add_argument(
        "opts",
        help="see config/s3dis/s3dis_stratified_transformer.yaml for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def boreas_save_label_file(label, i, data_root, gpu_number):
    label_paths = []
    with open("dataset/boreas_nuscenes.yaml", "r") as stream:
        boreas_nuscenes_dict = yaml.safe_load(stream)
    name = boreas_nuscenes_dict["name"]  # "boreas"
    for sequencce in name:
        dir = os.path.join(
            data_root, sequencce, "label"
        )  
        if not os.path.exists(dir):
            os.mkdir(dir)
        lidar_root = os.path.join(data_root, sequencce, "lidar")
        lidar_names = get_file_names(lidar_root)
        for j in range(len(lidar_names)):
            lidar_name = lidar_names[j]
            label_path = os.path.join(dir, str(lidar_name) + ".label")
            label_paths.append(label_path)
    label = label.astype(np.int32)
    if i * gpu_number + torch.cuda.current_device() < len(label_paths):
        save_path = label_paths[i * gpu_number + torch.cuda.current_device()]
        print("save label to {}".format(save_path))
        label.tofile(save_path)


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
        args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0
    )


def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(
            main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args)
        )
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # get model
    if args.arch == "unet_spherical_transformer":
        from model.unet_spherical_transformer import Semantic as Model

        args.patch_size = np.array(
            [args.voxel_size[i] * args.patch_size for i in range(3)]
        ).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = Model(
            input_c=args.input_c,
            m=args.m,
            classes=args.classes,
            block_reps=args.block_reps,
            block_residual=args.block_residual,
            layers=args.layers,
            window_size=window_size,
            window_size_sphere=window_size_sphere,
            quant_size=window_size / args.quant_size_scale,
            quant_size_sphere=window_size_sphere / args.quant_size_scale,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            window_size_scale=args.window_size_scale,
            grad_checkpoint_layers=args.grad_checkpoint_layers,
            sphere_layers=args.sphere_layers,
            a=args.a,
        )
    else:
        raise Exception("architecture {} not supported yet".format(args.arch))

    if main_process():
        global logger, writer
        logger = get_logger(args.save_path)
        logger.info(args)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.sync_bn:
            if main_process():
                logger.info("use SyncBN")
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model.cuda())

    if main_process():
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"], strict=True)

            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    collate_fn = partial(
        collate_fn_limit,
        max_batch_points=args.max_batch_points,
        logger=logger if main_process() else None,
    )

    args.use_tta = getattr(args, "use_tta", False)
    if args.data_name == "oxford_nuscenes":
        val_data = oxford_nuScenes(
            data_path=args.data_root,
            info_path_list=["nuscenes_seg_infos_1sweeps_val.pkl"],
            voxel_size=args.voxel_size,
            split="val",
            rotate_aug=args.use_tta,
            flip_aug=args.use_tta,
            scale_aug=args.use_tta,
            transform_aug=args.use_tta,
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None),
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    elif args.data_name == "boreas_nuscenes":
        val_data = boreas_nuScenes(
            data_path=args.data_root,
            info_path_list=["nuscenes_seg_infos_1sweeps_val.pkl"],
            voxel_size=args.voxel_size,
            split="val",
            rotate_aug=args.use_tta,
            flip_aug=args.use_tta,
            scale_aug=args.use_tta,
            transform_aug=args.use_tta,
            xyz_norm=args.xyz_norm,
            pc_range=args.get("pc_range", None),
            use_tta=args.use_tta,
            vote_num=args.vote_num,
        )
    else:
        raise ValueError("The dataset {} is not supported.".format(args.data_name))

    if main_process():
        logger.info("val_data samples: '{}'".format(len(val_data)))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False
        )
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        collate_fn=collation_fn_voxelmean,
    )
    if args.val:
        validate_distance(val_loader, model)
        exit()


def validate_distance(val_loader, model):
    if main_process():
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

    torch.cuda.empty_cache()

    model.eval()
    for i, batch_data in enumerate(val_loader):
        (coord, xyz, feat, target, offset, inds_reverse) = batch_data
        inds_reverse = inds_reverse.cuda(non_blocking=True)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()

        coord = torch.cat([batch.unsqueeze(-1), coord], -1)
        spatial_shape = np.clip((coord.max(0)[0][1:] + 1).numpy(), 128, None)

        coord, xyz, feat, target, offset = (
            coord.cuda(non_blocking=True),
            xyz.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)

        sinput = spconv.SparseConvTensor(
            feat, coord.int(), spatial_shape, args.batch_size_val
        )

        assert batch.shape[0] == feat.shape[0]

        with torch.no_grad():
            output = model(sinput, xyz, batch)
            output = output[inds_reverse, :]

        output = output.max(1)[1]
        output = output.cpu().numpy()
        boreas_save_label_file(output, i, args.data_root, len(args.train_gpu))
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    import gc

    gc.collect()
    main()
