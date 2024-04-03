#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

"""
Code for preprocess MLS dataset:
https://www.openslr.org/94/
"""

import os
import sys
import shutil
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.file_utils import json_dump
from utils.audio_utils import audio_to_flac

def read_txt(txt_file):
    with open(txt_file) as f:
        lines = [line.rstrip() for line in f]
    return lines
  
if __name__ == '__main__':

    dataset_name = 'mls_german'
    data_dir = os.path.join('/mnt', dataset_name) 
    output_dir = os.path.join('/mnt/processed_datasets', dataset_name)
    AUDIO_SAVE_SAMPLE_RATE = 16000

    # transfer "train, dev, test, train-noisy" to "train, test, valid".
    split_output_name_dict = {
        'train': 'train',
        'dev': 'valid',
        'test': 'test',
    }

    file_ids = {'train': 1, 'dev': 1, 'test': 1}
    
    print(f"Processing {dataset_name} dataset...\n")
    
    for split in split_output_name_dict.keys():
        print(f"\n--> {split} split <--.\n")
        os.makedirs(os.path.join(output_dir, split_output_name_dict[split]), exist_ok = True)
        segments_dict = {}

        segments_file = os.path.join(data_dir, split, 'segments.txt')
        transcripts_file = os.path.join(data_dir, split, 'transcripts.txt')

        segments = read_txt(segments_file)
        transcripts = read_txt(transcripts_file)

        for segment, transcript in tqdm(zip(segments, transcripts), total=len(transcripts)):
            
            s_id, link_to_file,  segment_start, segment_end = segment.split()
            t_id , trans = transcript.split("\t", 1)

            # Check if information is of the same file
            if s_id == t_id:
                id = s_id
                
            else:
                print(f'Data discrepancy: {s_id} != {t_id}')
                continue
            
            # Get speaker id
            speaker_id, book_id, file_id = id.split('_')

            audio_path = os.path.join(data_dir, split, 'audio', speaker_id, book_id, id)
            audio_output_path = os.path.join(output_dir, split_output_name_dict[split], f'{file_ids[split]}.flac')
            json_output_path = os.path.join(output_dir, split_output_name_dict[split], f'{file_ids[split]}.json')

            # convert to flac here if the input is wav
            if os.path.isfile(audio_path + '.flac'):
                audio_path = audio_path + '.flac'
                shutil.copyfile(audio_path, audio_output_path)
            
            else:
                audio_to_flac(audio_path, audio_output_path, sample_rate = AUDIO_SAVE_SAMPLE_RATE)

            audio_json = {
                "text": [trans],
                "tag": ['MLS', 'German', 'Speech'],
                "original data": {  # Metadata
                    "title": "MLS Dataset",
                    "link_to_file": link_to_file,
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "speaker_id": speaker_id,
                    "book_id": book_id,
                    "file_id": file_id,
                    "id": id,
                    }
            }
            
            json_dump(audio_json, json_output_path)
            file_ids[split] += 1