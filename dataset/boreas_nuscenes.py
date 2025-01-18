import os
import random
import numpy as np
import torch
import yaml
import pickle
import glob
from pathlib import Path
from os.path import join, exists
from util.data_util import data_prepare
from tool.lidar import get_file_names


class boreas_nuScenes(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        info_path_list,
        voxel_size=[0.1, 0.1, 0.1],
        split="train",
        return_ref=True,
        label_mapping="dataset/boreas_nuscenes.yaml",
        rotate_aug=True,
        flip_aug=True,
        scale_aug=True,
        transform_aug=True,
        trans_std=[0.1, 0.1, 0.1],
        ignore_label=255,
        voxel_max=None,
        xyz_norm=False,
        pc_range=None,
        use_tta=None,
        vote_num=4,
    ):
        super().__init__()
        self.return_ref = return_ref
        self.split = split
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.transform_aug = transform_aug
        self.trans_std = trans_std
        self.ignore_label = ignore_label
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm
        self.pc_range = None if pc_range is None else np.array(pc_range)
        self.data_path = data_path
        self.class_names = [
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
        ]
        self.use_tta = use_tta
        self.vote_num = vote_num
        self.lidar_paths = []
        self.label_paths = []

        with open("dataset/boreas_nuscenes.yaml", "r") as stream:
            self.nuscenes_dict = yaml.safe_load(stream)

        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)
        self.voxel_size = voxel_size

        name = self.nuscenes_dict["name"]  # "boreas"
        for sequencce in name:
            lidar_dir_name = "lidar"
            lidar_dir = os.path.join(self.data_path, sequencce, lidar_dir_name)
            lidar_names = get_file_names(lidar_dir)
            for i in range(len(lidar_names)):
                self.lidar_paths.append(
                    os.path.join(lidar_dir, str(lidar_names[i]) + ".bin")
                )

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.lidar_paths)

    def __getitem__(self, index):
        return self.get_single_sample(index)

    def get_single_sample(self, index, vote_idx=0):
        lidar_path = self.lidar_paths[index]
        pc = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 6))
        points = pc[:, :4].copy()
        labels_in = np.zeros(points.shape[0]).astype(np.uint8)

        if self.return_ref:
            feats = points[:, :4]
        else:
            feats = points[:, :3]
        xyz = points[:, :3]

        if self.pc_range is not None:
            xyz = np.clip(xyz, self.pc_range[0], self.pc_range[1])

        coords, xyz, feats, labels, inds_reconstruct = data_prepare(
            xyz,
            feats,
            labels_in,
            self.split,
            self.voxel_size,
            self.voxel_max,
            None,
            self.xyz_norm,
        )
        return (
            coords,
            xyz,
            feats,
            labels,
            inds_reconstruct,
        )
