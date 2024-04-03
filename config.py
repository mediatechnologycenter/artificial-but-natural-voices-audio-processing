#  SPDX-License-Identifier: MIT
#  © 2024 ETH Zurich, see AUTHORS.txt for details

from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    supported_audio_formats: tuple = ("mp3", "wav", "ogg", "flac")
