#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from tqdm import tqdm
import os
import subprocess
from pyannote.audio import Pipeline
import torch


class SpeakerSplitter:
    def __init__(self) -> None:
        access_token = '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'
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                            use_auth_token=access_token)
    
    def split_audio_improved(self, f_name: str, diarization: torch.Tensor, out_path: str) -> None:
        prev = None
        interval = [None, None]

        for turn, _, speaker in diarization.itertracks(yield_label=True):

            if prev is None:
                prev = speaker
                interval = [turn.start, turn.end]
            

            if prev != speaker:

                tmp_path = os.path.join(out_path, prev)
                os.makedirs(tmp_path, exist_ok=True)
                curr = len(os.listdir(tmp_path))
                command = f"ffmpeg -hide_banner -loglevel error -ss {interval[0]:.1f} -t {interval[1] - interval[0]:.1f} -i {f_name} {os.path.join(tmp_path, str(curr) + '.wav')}"


                # os.makedirs(out_path, exist_ok=True)
                # curr = len(os.listdir(out_path))
                # command = f"ffmpeg -hide_banner -loglevel error -ss {interval[0]:.1f} -t {interval[1] - interval[0]:.1f} -i {f_name} {os.path.join(out_path, str(curr) + '.wav')}"

                subprocess.call(command, shell=True)

                prev = speaker
                interval = [turn.start, turn.end]
            
            else:
                interval[1] = turn.end
        
        tmp_path = os.path.join(out_path, speaker)
        os.makedirs(tmp_path, exist_ok=True)
        curr = len(os.listdir(tmp_path))
        command = f"ffmpeg -hide_banner -loglevel error -ss {interval[0]:.1f} -t {interval[1] - interval[0]:.1f} -i {f_name} {os.path.join(tmp_path, str(curr) + '.wav')}"


        # os.makedirs(out_path, exist_ok=True)
        # curr = len(os.listdir(out_path))

        # command = f"ffmpeg -hide_banner -loglevel error -ss {interval[0]:.1f} -t {interval[1] - interval[0]:.1f} -i {f_name} {os.path.join(out_path, str(curr) + '.wav')}"

        subprocess.call(command, shell=True)

    def split_audio(self, f_name: str, diarization: torch.Tensor, out_path: str) -> None:
    

        for turn, _, speaker in diarization.itertracks(yield_label=True):

            tmp_path = os.path.join(out_path, speaker)
            os.makedirs(tmp_path, exist_ok=True)
            curr = len(os.listdir(tmp_path))
            command = f"ffmpeg -hide_banner -loglevel error -ss {turn.start:.1f} -t {turn.end - turn.start:.1f} -i {f_name} {os.path.join(tmp_path, str(curr) + '.wav')}"

            subprocess.call(command, shell=True)

    
    def __call__(self, f_name: str, split_audio: bool, out_path: str) -> torch.Tensor:
        '''
        f_name: path to audio file,
        split_audio: flag, if true split the audio file
        out_path: path to the output folder
        '''

        # run pipeline
        diarization = self.pipeline(f_name)

        # dump the diarization output to disk using RTTM format
        with open(os.path.join(out_path, "diarization.rttm"), "w") as rttm:
            diarization.write_rttm(rttm)

        if split_audio:
            self.split_audio(f_name, diarization, out_path)
        

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Pyannote Argument Parser.')

    parser.add_argument('in_path', type=str, help='Path to the input folder.')
    parser.add_argument('out_path', type=str, help='Path to the input folder.')

    args = parser.parse_args()

    # data path 
    in_path = args.in_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    # make pyannote model
    pipeline = SpeakerSplitter()

    for f_name in tqdm([os.path.join(in_path, f) for f in os.listdir(in_path)]):

        split_out = out_path + f_name.split('/')[-1].split('.')[-2]
        os.makedirs(split_out, exist_ok=True)

        print('Splitting file: ', f_name)
        pipeline(f_name, True, split_out)

    

    
