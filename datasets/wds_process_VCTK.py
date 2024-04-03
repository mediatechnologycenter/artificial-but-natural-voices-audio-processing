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

    dataset_name = 'VCTK-Corpus'
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
    num_sample = {'train': 1, 'dev': 1, 'test': 1}
    
    print(f"Processing {dataset_name} dataset...\n")
    
    for split in split_output_name_dict.keys():
        print(f"\n--> {split} split <--\n")
        os.makedirs(os.path.join(output_dir, split_output_name_dict[split]), exist_ok = True)
        segments_dict = {}

        audios_path = os.path.join(data_dir, 'wav48')
        transcripts_path = os.path.join(data_dir, 'txt')

        people_list = set(os.listdir(audios_path)) & set(os.listdir(transcripts_path))

        for person in tqdm(people_list):
            audio_files = sorted([os.path.join(audios_path, person, f) for f in os.listdir(os.path.join(audios_path, person))])
            transcripts_files = sorted([os.path.join(transcripts_path, person, f) for f in os.listdir(os.path.join(transcripts_path, person))])

            # Split test: 10, val: 10, train:[20:]
            if split == 'test':
                audio_files = audio_files[:10]
                transcripts_files = transcripts_files[:10]
            
            elif split == 'dev':
                audio_files = audio_files[10:20]
                transcripts_files = transcripts_files[10:20]
            
            elif split == 'train':
                audio_files = audio_files[20:]
                transcripts_files = transcripts_files[20:]

            for audio_file, transcript_file in zip(audio_files, transcripts_files):
                
                t_name = transcript_file.split("/")[-1]
                t_name = t_name.split(".")[0]

                a_name = audio_file.split("/")[-1]
                a_name = a_name.split(".")[0]

                # Check if information is of the same file
                if t_name == a_name:
                    id = a_name
                    
                else:
                    print(f'Data discrepancy: {t_name} != {a_name}')
                    continue
                
                # Get speaker id
                speaker_id, file_id = id.split('_')

                audio_output_path = os.path.join(output_dir, split_output_name_dict[split], f'{file_ids[split]}.flac')
                json_output_path = os.path.join(output_dir, split_output_name_dict[split], f'{file_ids[split]}.json')

                # convert to flac here if the input is wav
                if os.path.isfile(audio_file.split('.')[0] + '.flac'):
                    audio_path = audio_path + '.flac'
                    shutil.copyfile(audio_path, audio_output_path)
                
                else:
                    audio_to_flac(audio_file, audio_output_path, sample_rate = AUDIO_SAVE_SAMPLE_RATE)

                # read transcript
                trans = read_txt(transcript_file)

                audio_json = {
                    "text": trans,
                    "tag": ['VCTK', 'German', 'Speech'],
                    "original data": {  # Metadata
                        "title": "VCTK Dataset",
                        "speaker_id": speaker_id,
                        "file_id": file_id,
                        "id": id,
                        }
                }
                
                json_dump(audio_json, json_output_path)
                file_ids[split] += 1