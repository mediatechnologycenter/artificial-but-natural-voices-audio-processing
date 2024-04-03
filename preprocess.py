#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import argparse
import os
import subprocess
import torch
from tqdm import tqdm

class Preprocessor:

    def __init__(self, input_path=None, dataset_name=None, output_path=None) -> None:

        self.update_paths(input_path, dataset_name, output_path)
        
        self.tasks = {
            'split_audio': lambda: self.split_audio(),
            'track_separation': lambda: self.separate_tracks(),
            'denoising': lambda: self.denoise(),
            'diarization': lambda: self.diarize(),
            'speaker_embedding': lambda: self.speaker_embed(),
            'transcription': lambda: self.transcribe(),
        }

        self.speaker_encoder_model_path = 'utils/encoder_models/best_model.pth.tar'
        self.speaker_encoder_config_path = 'utils/encoder_models/config.json'
    
    def update_paths(self, input_path=None, dataset_name=None, output_path=None):
        self.input_path = input_path
        self.dataset_name = dataset_name
        self.output_path = os.path.join(output_path, dataset_name)
        os.makedirs(self.output_path, exist_ok=True)

        self.dataset_path = os.path.join(input_path, dataset_name)

        # Initialize paths for output folders
        self.split_path = os.path.join(self.output_path, 'split')
        self.demucs_path = os.path.join(self.output_path, 'demucs')
        self.denoised_path = os.path.join(self.output_path, 'denoised')
        self.diarization_path = os.path.join(self.output_path, 'speech_separation')
        self.transcriptions_path = os.path.join(self.output_path, 'transcriptions')
    
    def get_wav(self,file_path):
        # Convert * to wav
        command = f"ffmpeg -hide_banner -loglevel error -y -i {file_path} -ab 160k -ac 2 -ar 44100 -vn {''.join(file_path.split('.')[:-1])}.wav"
        subprocess.call(command, shell=True)
    
    def split_audio(self):
        from preprocessing.audio_splitter import SplitWavAudioMubin
        print('\n--> Splitting audios in segments..\n')

        # Convert mp4 to wav
        for f_name in [os.path.join(self.dataset_path, f) for f in os.listdir(self.dataset_path) if f.split('.')[-1] != 'wav']:
            self.get_wav(f_name)    
        
        os.makedirs(self.split_path, exist_ok=True)
        
        for f_name in tqdm(os.listdir(self.dataset_path)):
            splitter = SplitWavAudioMubin(folder=self.dataset_path, filename=f_name, out_path=self.split_path)
            splitter.multiple_split(min_per_split=3)
    
    def separate_tracks(self):
        from preprocessing.demucs_funcs import separate
        print('\n--> Separating vocals and non-vocals..\n')

        os.makedirs(self.demucs_path, exist_ok=True)

        # data path 
        separate(self.split_path, self.demucs_path)
    
    def denoise(self):
        from preprocessing.denoise_funcs import denoise
        import torchaudio

        print('\n--> Denoising audio signal..\n')

        # data path 
        in_path = os.path.join(self.demucs_path, 'htdemucs')
        if not os.path.exists(in_path):
            in_path = self.demucs_path

        os.makedirs(self.denoised_path, exist_ok=True)
        out_path = self.denoised_path


        for f_name in tqdm([os.path.join(in_path, f, 'vocals.wav') for f in os.listdir(in_path)]):

            denoised_out = os.path.join(out_path, f_name.split('/')[-2] + ".wav")

            original, denoised, sr = denoise(f_name)

            torchaudio.save(denoised_out, denoised.cpu(), sr)
    
    def diarize(self):
        from preprocessing.pyannote_funcs import SpeakerSplitter

        print('\n--> Diarize speakers..\n')

        # data path 
        in_path = self.denoised_path

        os.makedirs(self.diarization_path, exist_ok=True)

        out_path = self.diarization_path

        # make pyannote model
        pipeline = SpeakerSplitter()

        for f_name in tqdm([os.path.join(in_path, f) for f in os.listdir(in_path)]):
            
            split_out = os.path.join(out_path, f_name.split('/')[-1].split('.')[-2])

            os.makedirs(split_out, exist_ok=True)
            # print('Splitting file: ', f_name)
            pipeline(f_name, True, split_out)

    def speaker_embed(self):
        from TTS.config import load_config
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.utils.managers import save_file
        from preprocessing.compute_embeddings import SpeakerEncoder

        from datasets.dataset_formatters import tmp_formatter, VCTK_formatter


        print('\n--> Compute speaker embeddings..\n')

        self.speaker_embedding_path = os.path.join(self.output_path, 'speakers.pth')

        class dotdict(dict):
            """dot.notation access to dictionary attributes"""
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        args = {
            'model_path' : self.speaker_encoder_model_path,
            'config_path' : self.speaker_encoder_config_path,
            'config_dataset_path' : None,
            'output_path' : self.speaker_embedding_path,
            'old_file' : None,
            'no_eval': True,
            'formatter_name' : False,
            'dataset_name' : 'speech_separation',
            'dataset_path' : self.diarization_path,
            'metafile' : None,
        }

        args = dotdict(args)

        use_cuda = torch.cuda.is_available()

        if args.config_dataset_path is not None:
            c_dataset = load_config(args.config_dataset_path)
            meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not args.no_eval)
        else:
            c_dataset = BaseDatasetConfig()
            c_dataset.dataset_name = args.dataset_name
            c_dataset.path = args.dataset_path
            c_dataset.meta_file_train = args.metafile if args.metafile else None
            if args.dataset_name == 'VCTK-Corpus':
                meta_data_train, meta_data_eval = load_tts_samples(c_dataset, formatter=VCTK_formatter, eval_split=not args.no_eval)

            else:
                meta_data_train, meta_data_eval = load_tts_samples(c_dataset, formatter=tmp_formatter, eval_split=not args.no_eval)


        if meta_data_eval is None:
            samples = meta_data_train
        else:
            samples = meta_data_train + meta_data_eval

        encoder_manager = SpeakerEncoder(model_path=args.model_path,
                                        config_path=args.config_path,
                                        use_cuda=use_cuda)

        # compute speaker embeddings
        speaker_mapping = {}

        for idx, fields in enumerate(tqdm(samples)):
            
            root_path = fields["root_path"]
            track_name = fields["track_name"]
            file_name = fields["recording_name"]
            speaker_id = fields["speaker_id"]
            audio_file = fields["audio_file"]
    
            # extract the embedding
            try:
                embedd = encoder_manager.load_and_embed(audio_file)
            
            except:
                print('Something wrong with: ', audio_file)
                continue

            # create speaker_mapping if target dataset is defined
            if args.dataset_name == 'VCTK-Corpus':
                speaker_mapping[track_name] = {}
                speaker_mapping[track_name] = embedd
            
            else:
                speaker_mapping[file_name + '/' + speaker_id + '/' + track_name] = {}
                speaker_mapping[file_name + '/' + speaker_id + '/' + track_name] = embedd

        print('Number of speakers: ', len(speaker_mapping.items()))

        if speaker_mapping:
            # save speaker_mapping if target dataset is defined
            if os.path.isdir(args.output_path):
                mapping_file_path = os.path.join(args.output_path, "speakers.pth")
            else:
                mapping_file_path = args.output_path

            if os.path.dirname(mapping_file_path) != "":
                os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

            save_file(speaker_mapping, mapping_file_path)
            print("Speaker embeddings saved at:", mapping_file_path)

    def transcribe(self):
        from transcriber.transcribe import transcribe
        print('\n--> Transcribe all clips..\n')
        input_path = self.diarization_path
        transcribe(dataset_path=input_path, output_path=self.transcriptions_path, from_diarization=True)

    def __call__(self, task):
        self.tasks[task]()
        torch.cuda.empty_cache()
    
    def run_all_steps(self):
        for task in self.tasks:
            self.tasks[task]()
            torch.cuda.empty_cache()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Preprocessor arguments.')

    parser.add_argument('input_path', type=str, help='Path to the input folder.')
    parser.add_argument('dataset_name', type=str, help='Dataset name.')
    parser.add_argument('output_path', type=str, help='Path to the output folder.')
    parser.add_argument('task', type=str, help='Nome of the task to execute. ("all" to execute all tasks')

    args = parser.parse_args()

    # print(F'\nRUNNING PREPROCESSING on {args.dataset_name} dataset..\n\n')

    # make preprocessor
    preprocessor = Preprocessor(args.input_path, args.dataset_name, args.output_path)

    # run the task
    if args.task == 'all':
        preprocessor.run_all_steps()

    else:
        preprocessor(args.task) 
