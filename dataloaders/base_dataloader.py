#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import webdataset as wds
import soundfile as sf
import io
import os
from tqdm import tqdm
import logging
import json
import math

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

def preprocess(
    sample,
):
    """
    Preprocess a single sample for wdsdataloader.
    """
    audio_ext = "flac"
    text_ext = "json"
    audio_data, orig_sr = sf.read(io.BytesIO(sample[audio_ext]))
    json_dict_raw = json.loads(sample[text_ext].decode("utf-8"))
    sample["waveform"] = audio_data
    texts = json_dict_raw["text"]
    metadata = json_dict_raw["original data"]

    sample["text"] = texts
    sample["metadata"] = metadata
    return sample


class BaseDataloader():
    def __init__(self, dataset_path, split, batch_size=1, shuffle=False) -> None:
        input_shards = [os.path.join(dataset_path, split, f) for f in os.listdir(os.path.join(dataset_path, split)) if os.path.join(dataset_path, split, f).endswith('.tar')]

        # Load sizes file
        with open(os.path.join(dataset_path, split, 'sizes.json'), 'r') as j:
            sizes_file = json.loads(j.read())
            self.total_elements = sum(sizes_file[f] for f in sizes_file) 
            self.size = math.ceil(self.total_elements/batch_size)

        # make pipeline to load data
        pipeline = [wds.SimpleShardList(input_shards)]

        pipeline.extend(
            [
                wds.split_by_node,
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=log_and_continue),
                wds.map(preprocess),
                self.data_tuple(),
                wds.batched(1),
            ]
        )
        self.dataset = wds.DataPipeline(*pipeline)
        self.dataloader = wds.WebLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    
    def process(self, sample):
        
        """
        Preprocess a single sample for wdsdataloader.
        """
        audio_ext = "flac"
        text_ext = "json"
        audio_data, orig_sr = sf.read(io.BytesIO(sample[audio_ext]))
        json_dict_raw = json.loads(sample[text_ext].decode("utf-8"))
        sample["waveform"] = audio_data
        sample["text"] = json_dict_raw["text"]

        return sample
    
    def data_tuple(self):
        '''Modify the return tumple for the specific dataset'''
        return wds.to_tuple("__url__", "__key__", "waveform", "text")

    def __len__(self):
        return self.size