#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from glob import glob
import json
from pathlib import Path
import statistics as st
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="SRF Prepare Transcriptions")
    parser.add_argument(
        "--path", type=str, help="Path to SRF dataset"
    )
    

    args = parser.parse_args()
    return args


def parse_srf_tranriptions(path: str, save_to_csv: bool =False):
    """
    Get all transcriptions files for an SRF dataset and return a tuple id, text, average confidence of given transcription
    computed as the average of the confidence for each of the tokens.

    Arguments:

    path (Required): Path to root directory of the SRF dataset
    save_to_csv (Optional): Whether to save the results to a file named `metadata.csv` under the dataset path. Default False.
    """
    files = glob(str(Path(path,"**/transcripts/*.json")))

    print(f"Found {len(files)} files in {path}")

    processed_tuples = []
    for file in tqdm(files):

        with open(file, 'r') as f:
            data = f.read()
        obj = json.loads(data)
        results = obj['results']
        # print(results)

        # filename = Path(file).stem
        filename = file.split(path)[1][1:]

        contents = []
        confidences = []

        # There are some files with no transcription because they do not have people speaking, such as SRFC009341_29965AC2BF6548349362967A8D30A511.WAV
        # Skip those for now
        if not results:
            continue

        for res in results:
            # We get the most likely alternative
            alternative = res["alternatives"][0]

            # Quite hacky, but works?
            if "attaches_to" in res:
                pass
            else:
                contents.append(' ')

            contents.append(alternative["content"])
            confidences.append(alternative["confidence"])
        
        sentence = ''.join(contents)[1:] # Remove space at the beginning
        
        # print(filename,sentence)
        
        processed_tuples.append((filename, sentence, st.mean(confidences)))

    if save_to_csv:
        import csv
        with open(Path(path, "metadata.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(processed_tuples)
    return processed_tuples


if __name__ == "__main__":

    args = parse_args()
    result = parse_srf_tranriptions(path=args.path, save_to_csv=True)
    # print(result)
