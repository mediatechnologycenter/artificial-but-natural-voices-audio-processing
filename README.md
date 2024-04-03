# Audio Processing 

You can find here a complete audio processing pipeline designed to prepare data for TTS applications.

You can use this code to transform a collection of audio/video material into an audio dataset suitable for training.

## Pipeline structure

The pipeline follows the following structure:
- [Audio track separation](https://github.com/facebookresearch/demucs): extracts the audio signals and splits them into vocals and non-vocals. 
- [Audio Denoising](https://github.com/facebookresearch/denoiser#real-time-speech-enhancement-in-the-waveform-domain-interspeech-2020): enhances and denoises the vocal signals.
- [Speaker Diarization](https://github.com/pyannote/pyannote-audio): detects different speakers thorughout the audio tracks. 
- [Speaker Embedding](https://github.com/coqui-ai/TTS/tree/dev/TTS/encoder): computes speaker embeddings across all audios.
- [Speaker Identification](): speakers are clustered to get speakers identities across all audio data. 

## Requirements

Multiple environments are necessary across the pipeline.

### DEMUCS  

```
conda create -y -n demucs python=3.7
conda activate demucs
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install -U git+https://github.com/facebookresearch/demucs#egg=demucs
pip install tqdm
conda update -y ffmpeg
pip install pyannote.audio
```

### DENOISE + PYANNOTE (SPEAKER DIARIZATION)   

```
conda create -y -n denoiser python=3.8
conda activate denoiser
# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install denoiser
pip install tqdm

```

### TTS

```
conda create -y -n tts python=3.8
conda activate tts
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install TTS
pip install dynamicviz

```

## How to run

In order to run the pipeline over a collection of files:
```
bash process_audio.sh INPUTDATAPATH DATASETNAME OUTPUTPATH
```
Where: 
- INPUTDATAPATH: path to the data folder.
- DATASETNAME: name of the dataset to process.
- OUTPUTDATAPATH: path to the output folder.

Example:
```
bash process_audio.sh /home/alberto/NaturalVoices data/ /home/alberto/NaturalVoices/results
```

## How to run on VCTK dataset
Download the dataset from: https://datashare.is.ed.ac.uk/handle/10283/2651 .
Unpack the dataset and run the following:
```
bash process_VCTK.sh INPUTDATAPATH
```
Where: 
- INPUTDATAPATH: path to the data folder cointaining the unpacked VCTK dataset.

Example:
```
bash process_VCTK.sh /mnt
```

Take a look at the "visualize_embeddings_VCTK.ipynb" to visualize the embeddings.
<br/><br/>
# WebDataset

Here are some intructions to build a webdataset. \
Webdatasets are used to easily transfer and load data. By standardizing the dataset structure, we can train multiple models with multiple dataset with less confusion and errors. \

In order to build a webdataset follow these instructions:
- **Process the dataset**: make a dataset specific script to build .flac - .json pairs. See examples in `/datasets`. \
Example Usage: `python wds_process_VCTK.py`
- **Compress the dataset**: compress the dataset using `/utils/make_tar.py`. \
Example Usage: `python make_tar.py --input /mnt/processed_datasets/mls_german/ --output /mnt/webdataset_tar/mls_german/`
- **Load the dataset**: test the new dataset by loading it with `/utils/wds_test_tars.py`. \
Example Usage: `python wds_test_tars.py --tar-path /mnt/webdataset_tar/VCTK-Corpus/train/`

## DataLoaders

You can now load the Webdataset using specialised dataloaders. \
Inherit from `BaseDataloader` and make your own in `/dataloaders`.
Overwrite the `process` and `data_tuple` methods to extract the right information from your data.