#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from .base_dataloader import BaseDataloader
import soundfile as sf
import io
import json
import webdataset as wds

class VCTK_dataloader(BaseDataloader):
    def __init__(self, dataset_path, split, batch_size=1, shuffle=False) -> None:
        super().__init__(dataset_path, split, batch_size, shuffle)

    def process(self, sample):
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
    
    def data_tuple(self):
        '''Modify the return tumple for the specific dataset'''
        return wds.to_tuple("__url__", "__key__", "waveform", "text", "metadata")