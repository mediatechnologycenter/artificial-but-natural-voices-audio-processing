#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import argparse
import os
import json
import whisper
import numpy as np
from tqdm import tqdm
import glob
from pathlib import Path
from stable_whisper import tighten_timestamps, group_word_timestamps
from whisper.normalizers import BasicTextNormalizer

normalizer = BasicTextNormalizer()
# from stable_whisper import modify_model
from confidence_score_patch import modify_model
from naturstimmen.config import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', help='Dataset path.')
    parser.add_argument('--model_size', help='Model size.', default="large-v2")
    parser.add_argument('--language', help='Language in audio.', default=None)
    parser.add_argument('--word_level', help='Generate timestamps at word level instead of segment level.', default=False)
    args = parser.parse_args()
    return args



def results_to_sentence_level(whisper_results):
    """
    end_at_last_word: bool
        set end-of-segment to timestamp-of-last-token
    end_before_period: bool
        set end-of-segment to timestamp-of-last-non-period-token
    start_at_first_word: bool
        set start-of-segment to timestamp-of-first-token
    """
    segs = tighten_timestamps(whisper_results,
                        end_at_last_word=False,
                        end_before_period=False,
                        start_at_first_word=False)['segments']
    max_idx = len(segs) - 1
    i = 1
    while i <= max_idx:
        if not (segs[i]['end'] - segs[i]['start']):
            if segs[i - 1]['end'] == segs[i]['end']:
                segs[i - 1]['text'] += (' ' + segs[i]['text'].strip())
                del segs[i]
                max_idx -= 1
                continue
            else:
                segs[i]['start'] = segs[i - 1]['end']
        i += 1
    return segs

def results_to_word_level(whisper_results):
    """

    Parameters
    ----------
    res: dict
        results from modified model
    srt_path: str
        output path of srt
    combine_compound: bool
        concatenate words without inbetween spacing
    strip: bool
        perform strip() on each word
    min_dur: bool
        minimum duration for each word (i.e. concat the words if it is less than specified value; Default 0.02)

    """
    return group_word_timestamps(whisper_results, combine_compound=False, min_dur=None)

def clean_segments(segs):
    return [{"text": sub["text"].strip(),
           "start":sub["start"],
            "end": sub["end"],
            "whole_word_timestamps": sub["whole_word_timestamps"]} for sub in segs]

def postprocess_segments(whisper_results, word_level:bool =False):
    
    if word_level:
        segs = results_to_word_level(whisper_results)
    else:
        segs= results_to_sentence_level(whisper_results)
        
    return clean_segments(segs=segs)

def transcribe(dataset_path:str, output_path:str=None, model_size:str = "large-v2", audio_language:str = None, word_level:bool=False, from_diarization:bool = False) -> None:
    """
    Creates trasncriptions files

    Params:
    - dataset_path: Root directory containing the audios to be transcribed
    - output_path: Output directory. If not provided, a `transcriptions` directory will be created under dataset_path
    - model_size: Whisper model size, `large-v2` by default. Recommended to use `model_size=small` for debugging.
    - audio_language: Language of the audios. if not provided, it will be autodetected.
    - word_level: If True, provides timing at the word level.
    - from_diarization: If true, the expected input directory is expected to have the structure produced in the diarization step.
    """
    if not output_path:
       output_path = os.path.join(dataset_path, "transcriptions") 

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    audio_transcriptions_folder = os.path.join(output_path, "audio_transcriptions")
    if not os.path.exists(audio_transcriptions_folder):
        os.mkdir(audio_transcriptions_folder)
    
    audios = []
    for format in Config.supported_audio_formats:
        audios.extend(glob.glob(os.path.join(dataset_path, f"**/*.{format}"),recursive=True))

    print(f"Found {len(audios)} audios to transcribe.")



    model = whisper.load_model(model_size)
    modify_model(model)

    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )
    

    for audio in tqdm(audios):
        if from_diarization:
            path = Path(audio)
            name = f"{path.parent.name}_{path.stem}"
        else:
            name = Path(audio).stem

        try:
            results = model.transcribe(audio,language=audio_language)
            segments = postprocess_segments(results, word_level=word_level)
            entry = {  "path": audio.split(dataset_path)[-1],
                            "text": results["text"],
                            "language": results["language"],
                            "segments": segments
                    }
                    

            with open(os.path.join(audio_transcriptions_folder,f"{name}.json"), 'w', encoding='utf-8') as f:
                json.dump(entry, f, ensure_ascii=False, indent=4)
        
        except Exception as e:
            print(f"Couldn't transcribe the audio: {audio}. Exception: {e}.")    

    print("Creating transcriptions file")
    data = []
    for transcription_file in tqdm(glob.glob(audio_transcriptions_folder + "/*.json")):
        with open(transcription_file, 'r', encoding='utf-8') as f:
                transcription = json.load(f)
                data.append({
                    transcription["path"] : transcription["text"]
                })
    with open(os.path.join(output_path,"transcriptions.json"), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    args = parse_args()
    transcribe(dataset_path=args.dataset_path, model_size=args.model_size, audio_language=args.language, word_level=args.word_level)