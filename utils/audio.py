# Based on https://github.com/openai/whisper/blob/main/whisper/audio.py

from typing import Tuple
import ffmpeg
import numpy as np
import os
import torch
import torchaudio
import matplotlib.pyplot as plt


def get_audio_sampling_rate(file: str, metadata: bool = False) -> int:
    """
    Return the sampling rate of a given audio file. If metadata (default=`False`) flag is `True`, returns also the metadata object.
    """
    data = torchaudio.info(file)
    if metadata:
        return data.sample_rate, data
    return data.sample_rate

def load_audio(file: str, start:int = 0, duration:int = -1) -> Tuple[torch.Tensor, int]:
    """
    Open an audio file
    Parameters
    ----------
    file: str
        The audio file to open
    start: int
        The number of seconds to offset the audio loading. -1 (default) reads all the remaining samples.
    duration: int
        The number of seconds to load.
    Returns
    -------
    A `torch.tensor` containing the audio waveform, in float32 dtype with shape [channel, time].
    """

    sr = get_audio_sampling_rate(file)

    if duration != -1:
        duration = duration*sr

    waveform, sample_rate = torchaudio.load(file, frame_offset=start*sr, num_frames=duration)
    return waveform, sample_rate


SAMPLE_RATE = 48000
def load_audio_as_mono(file: str, sr: int = SAMPLE_RATE, start:int = 0, end:int = None):
    """
    Open an audio file and read as mono waveform, resampling as necessary
    Parameters
    ----------
    file: str
        The audio file to open
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        if end:
            out, _ = (
                ffmpeg.input(file, threads=0, ss=start, to=end)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
        else:
            out, _ = (
                ffmpeg.input(file, threads=0, ss=start)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
            )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)