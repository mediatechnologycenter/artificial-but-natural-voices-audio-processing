#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from typing import List
import os
import torch
import glob
import whisper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def init_dataset(dataset_path:str) -> List[str]:
    return glob.glob(os.path.join(dataset_path, "*.wav"))[:6]


class SRF(torch.utils.data.Dataset):
    """
    A simple class to wrap SRF audio data and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, dataset_path,device=DEVICE):
        self.dataset_path = dataset_path
        self.dataset = init_dataset(dataset_path)
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio_path = self.dataset[item]
        audio = whisper.load_audio(audio_path)
        audio = torch.from_numpy(audio)
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return mel
