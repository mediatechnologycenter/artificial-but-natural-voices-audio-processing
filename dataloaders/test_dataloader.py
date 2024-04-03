#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from base_dataloader import BaseDataloader
from VCTK_dataloader import VCTK_dataloader
from MLS_dataloader import MLS_dataloader

from tqdm import tqdm

VCTK_train_dataloader = VCTK_dataloader('/mnt/webdataset_tar/VCTK-Corpus', split='train')
MLS_train_dataloader = MLS_dataloader('/mnt/webdataset_tar/mls_german', split='train')

for el in tqdm(VCTK_train_dataloader.dataloader, total=len(VCTK_train_dataloader)):
    continue

for el in tqdm(MLS_train_dataloader.dataloader, total=len(MLS_train_dataloader)):
    continue

