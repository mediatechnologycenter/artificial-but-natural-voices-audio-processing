#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

import os
import subprocess
import io
from pathlib import Path
import select
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
import argparse

def find_files(in_path):
    out = []

    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in ["wav"]: # ["mp3", "wav", "ogg", "flac"]:
            out.append(file)
    return out

def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

def separate(inp=None, outp=None, model='htdemucs') -> None:

    # Options for the output audio.
    mp3 = False
    mp3_rate = 320
    float32 = True  # output as float 32 wavs, unsused if 'mp3' is True.
    int24 = False    # output as int24 wavs, unused if 'mp3' is True.
    # You cannot set both `float32 = True` and `int24 = True` !!

    two_stems = "vocals"

    inp = inp 
    outp = outp
    cmd = ["python", "-m", "demucs", "-o", str(outp), "-n", model]
    if mp3:
        cmd += ["--mp3", f"--mp3-bitrate={mp3_rate}"]
    if float32:
        cmd += ["--float32"]
    if int24:
        cmd += ["--int24"]
    if two_stems is not None:
        cmd += [f"--two-stems={two_stems}"]

    files = [str(f) for f in find_files(inp)]
    if not files:
        print(f"No valid audio files in {in_path}")
        return
    # print("Going to separate the files:")
    # print('\n'.join(files))
    # print("With command: ", " ".join(cmd))
    
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Demucs Argument Parser.')

    parser.add_argument('in_path', type=str, help='Path to the input folder.')
    parser.add_argument('out_path', type=str, help='Path to the input folder.')

    args = parser.parse_args()

    # data path 
    in_path = args.in_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)

    # RUN DEMUCS
    separate(in_path, out_path)