#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from pydub import AudioSegment
import math
import argparse
import os
import subprocess


class SplitWavAudioMubin():
    def __init__(self, folder, filename, out_path):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        self.out_path = out_path
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 60 * 1000
        t2 = to_min * 60 * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.out_path + '/' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 60)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demucs Argument Parser.')

    parser.add_argument('in_path', type=str, help='Path to the input folder.')
    parser.add_argument('out_path', type=str, help='Path to the input folder.')

    args = parser.parse_args()

    # data path 
    in_path = args.in_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)    

    # Convert mp4 to wav
    for f_name in [os.path.join(in_path, f) for f in os.listdir(in_path) if f.split('.')[-1] != 'wav']:

        command = f"ffmpeg -hide_banner -loglevel error -y -i {f_name} -ab 160k -ac 2 -ar 44100 -vn {''.join(f_name.split('.')[:-1])}.wav"

        subprocess.call(command, shell=True)    
    
    for f_name in os.listdir(in_path):

        splitter = SplitWavAudioMubin(folder=in_path, filename=f_name, out_path=out_path)
        splitter.multiple_split(min_per_split=3)


