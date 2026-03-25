'''
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

A simple PyTorch dataset using torchcodec for MP4 files and h5py for HDF5 files.
'''

import h5py
import numpy as np
import torch
from utils.data_utils import index_episodes
from torchcodec.decoders import VideoDecoder
from utils.skeleton_tfs import WRISTS

# loads only the wrist transforms by default. change as desired.
DEFAULT_QUERY_TFS = WRISTS

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, query_tfs=DEFAULT_QUERY_TFS):
        # a list of transforms that the dataset will return
        self.query_tfs = query_tfs

        # get episode paths and lengths
        self.dataset_path_list, self.episode_len = index_episodes(dataset_path)
        self.cumulative_len = np.cumsum(self.episode_len)

    def __len__(self):
        return sum(self.episode_len)
    
    def _locate_transition(self, index):
        # find a particular data point within an episode
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)  # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        return episode_index, start_ts
    
    def __getitem__(self, index):
        episode_id, frame_id = self._locate_transition(index)
        hdf5_file = self.dataset_path_list[episode_id]
        mp4_file = hdf5_file[:-5] + '.mp4'

        # grab info from HDF5
        with h5py.File(hdf5_file, "r") as root:
            tfdtype = root['/transforms/camera'][0].dtype 

            # get SE(3) transforms. Note: all transforms (including camera extrinsics) are expressed in the ARKit origin frame, 
            # which is a stationary frame on the ground that is set at the beginning of a recording session. 
            # the exact position and orientation of the origin frame depends on how the Vision Pro is initialized. 
            # you may want to instead express the transforms in the camera frame (see utils.data_utils.convert_to_camera_frame). 
            # you may also want to grab a "chunk" of N transforms with root['/transforms/'+tf_name][frame_id:frame_id+N] instead of just one. 
            tfs = np.zeros([len(self.query_tfs), 4, 4], dtype=tfdtype)
            for i, tf_name in enumerate(self.query_tfs):
                tfs[i] = root['/transforms/' + tf_name][frame_id]

            cam_ext = root['/transforms/camera'][frame_id] # extrinsics
            cam_int = root['/camera/intrinsic'][:] # intrinsics

            # natural language description of task
            if root.attrs['llm_type'] == 'reversible':
                direction = root.attrs['which_llm_description']
                lang_instruct = root.attrs['llm_description' if direction == '1' else 'llm_description2'] 
            else:
                lang_instruct = root.attrs['llm_description'] 

            # add joint prediction confidences, if present in this HDF5
            confs = None
            if 'confidences' in root.keys():
                confs = np.zeros([len(self.query_tfs)], dtype=tfdtype)
                for i, tf_name in enumerate(self.query_tfs):
                    confs[i] = root['/confidences/' + tf_name][frame_id]

        # grab image frame
        img = VideoDecoder(mp4_file, device='cpu')[frame_id]

        return tfs, cam_ext, cam_int, img, lang_instruct, confs