#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import argparse
import os
from argparse import RawTextHelpFormatter
import torch
import numpy as np
import math
from tqdm import tqdm

from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager

class SpeakerEncoder:
    def __init__(self, model_path, config_path, use_cuda):

        self.use_cuda = use_cuda

        self.encoder_manager = SpeakerManager(
            encoder_model_path=model_path,
            encoder_config_path=config_path,
            d_vectors_file_path=None,
            use_cuda=use_cuda,
        )

        self.sr = self.encoder_manager.encoder_ap.sample_rate
    
    def load_wavfile(self, audio_path):
        waveform = self.encoder_manager.encoder_ap.load_wav(audio_path, sr=self.sr)
        return waveform
    
    def embed(self, waveform):

        if not self.encoder_manager.encoder_config.model_params.get("use_torch_spec", False):
            m_input = self.encoder_manager.encoder_ap.melspectrogram(waveform)
            m_input = torch.from_numpy(m_input)
        else:
            m_input = torch.from_numpy(waveform)

        if self.use_cuda:
            m_input = m_input.cuda()

        m_input = m_input.unsqueeze(0)

        num_frames = 160 # each window is 25 ms -> 160*25/1000 = 4 seconds
        num_eval = int(m_input.shape[2]/num_frames)*2 # sliding window with 50% overlap
        embedding = self.encoder_manager.encoder.compute_embedding(m_input, num_frames, num_eval) 
        # TODO: had to change the source files at /home/alberto/anaconda3/envs/tts/lib/python3.8/site-packages/TTS/encoder/models/base_encoder.py

        return embedding

    def load_and_embed(self, audio_path):
        wavform = self.load_wavfile(audio_path)
        embedding = self.embed(wavform)
        return embedding

    def embed_until_time_t(self, waveform, time):

        end_time = int(time*self.sr)
        cut_input = waveform[:end_time]

        return self.embed(cut_input)
    
    def embed_across_time(self, wavform, granularity=4, verbose=False):

        # compute audio lenght
        audio_time = math.floor(wavform.shape[0] / self.sr)

        diff = []
        prev = self.embed_until_time_t(wavform, granularity)

        # show progress bar or not
        if verbose:
            rng = tqdm(np.arange(granularity*2, audio_time, granularity))
        else:
            rng = np.arange(granularity*2, audio_time, granularity)

        for i in rng:
            new = self.embed_until_time_t(wavform, i)
            diff.append(torch.mean(torch.abs(new-prev)).cpu().numpy())
            prev = new

        return np.stack(diff)
        

if __name__ == '__main__':
    from naturstimmen.datasets.dataset_formatters import tmp_formatter, VCTK_formatter

    parser = argparse.ArgumentParser(
        description="""Compute embedding vectors for each audio file in a dataset and store them keyed by `{dataset_name}#{file_path}` in a .pth file\n\n"""
        """
        Example runs:
        python TTS/bin/compute_embeddings.py --model_path speaker_encoder_model.pth --config_path speaker_encoder_config.json  --config_dataset_path dataset_config.json

        python TTS/bin/compute_embeddings.py --model_path speaker_encoder_model.pth --config_path speaker_encoder_config.json  --fomatter vctk --dataset_path /path/to/vctk/dataset --dataset_name my_vctk --metafile /path/to/vctk/metafile.csv
        """,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to model checkpoint file. It defaults to the released speaker encoder.",
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to model config file. It defaults to the released speaker encoder config.",
        default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json",
    )
    parser.add_argument(
        "--config_dataset_path",
        type=str,
        help="Path to dataset config file. You either need to provide this or `formatter_name`, `dataset_name` and `dataset_path` arguments.",
        default=None,
    )
    parser.add_argument("--output_path", type=str, help="Path for output `pth` or `json` file.", default="speakers.pth")
    parser.add_argument("--old_file", type=str, help="Previous embedding file to only compute new audios.", default=None)
    parser.add_argument("--disable_cuda", type=bool, help="Flag to disable cuda.", default=False)
    parser.add_argument("--no_eval", type=bool, help="Do not compute eval?. Default False", default=False)
    parser.add_argument(
        "--formatter_name",
        type=str,
        help="Name of the formatter to use. You either need to provide this or `config_dataset_path`",
        default=None,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset to use. You either need to provide this or `config_dataset_path`",
        default=None,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset. You either need to provide this or `config_dataset_path`",
        default=None,
    )
    parser.add_argument(
        "--metafile",
        type=str,
        help="Path to the meta file. If not set, dataset formatter uses the default metafile if it is defined in the formatter. You either need to provide this or `config_dataset_path`",
        default=None,
    )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.disable_cuda

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

    print(len(speaker_mapping.items()))

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