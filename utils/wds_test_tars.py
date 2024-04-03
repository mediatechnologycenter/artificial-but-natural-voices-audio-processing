import webdataset as wds
import soundfile as sf
import io
import os
import random
import copy
from tqdm import tqdm
import shutil
import argparse
import traceback
import logging
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tar-path",
        type=str,
        default=None,
        help="Path to the tars",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--order",
        default=False,
        action='store_true',
        help="if keep the search order accendingly",
    )
    args = parser.parse_args()
    return args

def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

def preprocess(
    sample,
):
    """
    Preprocess a single sample for wdsdataloader.
    """
    audio_ext = "flac"
    text_ext = "json"
    audio_data, orig_sr = sf.read(io.BytesIO(sample[audio_ext]))
    json_dict_raw = json.loads(sample[text_ext].decode("utf-8"))
    sample["waveform"] = audio_data
    texts = json_dict_raw["text"]
    metadata = json_dict_raw["original data"]

    sample["text"] = texts
    sample["metadata"] = metadata
    return sample


if __name__ == "__main__":
    args = parse_args()
    tar_path = args.tar_path
    input_shards = [os.path.join(tar_path, f) for f in os.listdir(tar_path) if os.path.join(tar_path, f).endswith('.tar')]
    pipeline = [wds.SimpleShardList(input_shards)]

    pipeline.extend(
        [
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.map(preprocess),
            wds.to_tuple("__url__", "__key__", "waveform", "text", "metadata"),
            wds.batched(1),
        ]
    )
    dataset = wds.DataPipeline(*pipeline)
    dataloader = wds.WebLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    old_k = 0
    old_batch = None
    try:
        for batch in tqdm(dataloader, total=len(input_shards)*512):
            continue

    except Exception as e:
        with open("check_tar_log.txt","a") as file:
            traceback.print_exc(file = file)
        print("old_batch:", old_batch)
