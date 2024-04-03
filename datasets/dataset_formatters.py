#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import os

def tmp_formatter(root_path, meta_file_train=None, ignored_speakers=None):
    
    items = []
    
    for folder in [os.path.join(root_path, f) for f in os.listdir(root_path)]:
        f_name = folder.split('/')[-1]

        for speaker in [os.path.join(folder, s) for s in os.listdir(folder)]:
            speaker_id = speaker.split('/')[-1]

            if speaker.endswith(".rttm"):
                    continue

            for w in os.listdir(speaker):
                
                wav_file = os.path.join(speaker, w)
                name = w.split('.')[0]

                print(f_name, speaker_id, name, wav_file)

                items.append(
                    {"audio_file": wav_file, 
                     "recording_name": f_name,
                     "speaker_id": speaker_id,
                     "track_name": name,
                     "root_path": root_path}
                )
    return items


def VCTK_formatter(root_path, meta_file_train=None, ignored_speakers=None):
    items = []
    
    for paragraph in [os.path.join(root_path, f) for f in os.listdir(root_path)]:
        speaker_id = paragraph.split('/')[-1]

        for speaker in [os.path.join(paragraph, s) for s in os.listdir(paragraph)]:
            p_name = speaker.split('/')[-1].split('_')[-1].split('.')[0]
            wav_file = speaker

            print(p_name, speaker_id, wav_file)

            items.append(
                    {"audio_file": wav_file, 
                     "speaker_id": speaker_id,
                     "recording_name": p_name,
                     "track_name": speaker.split('/')[-1].split('.')[0],
                     "root_path": root_path}
                )

    return items
