
#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio
import os
from tqdm import tqdm

# make model

def denoise(input_path):

    wav, sr = torchaudio.load(input_path)

    # model
    model = pretrained.dns64().cuda() 
    model.eval()

    # Convert to denoiser sample rate
    with torch.no_grad():
        wav_conv = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
        denoised = model(wav_conv[None])[0]
    
        # Convert back to original sample rate
        denoised = convert_audio(denoised, model.sample_rate, sr, model.chin)

    torch.cuda.empty_cache()
    return wav, denoised, sr


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Denoise Argument Parser.')

    parser.add_argument('in_path', type=str, help='Path to the input folder.')
    parser.add_argument('out_path', type=str, help='Path to the input folder.')

    args = parser.parse_args()

    # data path 
    in_path = args.in_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    # model
    denoiser = Denoiser()
    
    for f_name in tqdm([os.path.join(in_path, f, 'vocals.wav') for f in os.listdir(in_path)]):

        denoised_out = out_path + f_name.split('/')[-2] + ".wav"    

        original, denoised, sr = denoiser(f_name)

        torchaudio.save(denoised_out, denoised.cpu(), sr)