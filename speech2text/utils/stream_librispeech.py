# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import json
import argparse
import librosa
import soundfile as sf
import os
import numpy as np
from pydub.silence import split_on_silence

DURATION = 6.0
OVERLAP = 4.0
PAD_DURATION = 0.0
SR = 16000
#silence windows 200ms
WINDOW=200

def find_word_boundaries(text: str, start_time: float, end_time: float):
    """
    Estimate word boundary timestamps using uniform distribution
    Args:
        text: Text segment
        start_time: Start time of audio segment
        end_time: End time of audio segment
    Returns:
        List of estimated word boundary timestamps
    """
    words = text.split()
    duration = end_time - start_time
    # Assume uniform distribution of words across duration
    timestamps = [start_time + (i * duration / (len(words) - 1)) for i in range(len(words))]
    return timestamps

def find_chunk_boundary(audio_array: np.ndarray, sample_rate: int,
                       start_idx: int, target_idx: int, window: int = 200) -> int:
    """
    Find the nearest low-energy point to use as a chunk boundary
    Args:
        audio_array: Audio samples
        sample_rate: Sampling rate
        start_idx: Start index of search window
        target_idx: Target index for chunk boundary
        window: Search window size in samples
    Returns:
        Index of chosen boundary point
    """
    # Define search range
    search_start = max(start_idx, target_idx - window)
    search_end = min(len(audio_array), target_idx + window)

    # Calculate energy in small windows
    frame_length = int(0.02 * sample_rate)  # 200ms frames
    energies = []
    for i in range(search_start, search_end, frame_length):
        frame = audio_array[i:min(i + frame_length, search_end)]
        energy = np.sum(frame ** 2)
        energies.append((i, energy))

    # Find minimum energy point
    min_energy_idx = min(energies, key=lambda x: x[1])[0]
    return min_energy_idx


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    return args


def get_source_name(fname):
    basename_list, _ = os.path.splitext(fname)
    return "-".join(basename_list.split("-")[:2])


def get_sample_json(audio, transcript, fname):
    json_file = {
        "transcript": transcript,
        "files": [
            {
                "channels": 1,
                "sample_rate": float(SR),
                "bitdepth": 16,
                "bitrate": 256000.0,
                "duration": float(len(audio) / SR),
                "num_samples": int(len(audio)),
                "encoding": "Signed Integer PCM",
                "silent": False,
                "fname": fname,
                "speed": 1
            }
        ],
        "original_duration": float(len(audio) / SR),
        "original_num_samples": int(len(audio))
    }
    return json_file


def main():
    args = get_args()
    with open(args.manifest, "r") as manifest:
        json_data = json.load(manifest)

    os.makedirs(args.output_dir, exist_ok=True)

    pad_audio = np.zeros(int(PAD_DURATION * SR))

    catalog = dict()
    for data in json_data:
        original_fname = data["files"][0]["fname"]
        original_transcript = data["transcript"]
        original_audio = librosa.load(
            os.path.join(
                args.data_dir,
                original_fname),
            sr=SR)[0]
        original_json = get_sample_json(
            original_audio, original_transcript, original_fname)

        source_name = get_source_name(
            os.path.basename(
                os.path.basename(original_fname)))
        if source_name not in catalog:
            catalog[source_name] = []

        catalog[source_name].append((original_audio, original_json))

    full_json = []
    for key in catalog.keys():
        index = 0
        start = 0.0
        for entry in catalog[key]:
            clip_duration = entry[1]["original_duration"]
            full_transcript = entry[1]["transcript"]
            while start + 0.2 < (clip_duration - OVERLAP):
                end = min(start + DURATION, clip_duration)

                # Find actual chunk boundary near target_end
                if end < len(entry[0]):
                    chunk_end = float(find_chunk_boundary(entry[0], SR,  int(start * SR), int(end * SR))) / SR
                else:
                    chunk_end = end


                chunk = entry[0][int(start * SR):int(chunk_end * SR)]


                new_audio = np.concatenate([chunk, pad_audio])
                new_fname = os.path.join(
                    args.output_dir, key + "_" + str(index) + ".wav")
                new_json = get_sample_json(new_audio, full_transcript, new_fname)
                full_json.append(new_json)
                sf.write(new_fname, new_audio, SR)
                overlap_new_start = float(find_chunk_boundary(entry[0], SR, int((chunk_end - OVERLAP/2 - 0.5) * SR), int((chunk_end - OVERLAP/2 )*SR))) / SR
                start = overlap_new_start 
                index += 1

    # Creates json manifest containing all newly-repacked clips
    with open(args.output_json, "w") as manifest:
        json.dump(full_json, manifest, indent=2)

if __name__ == "__main__":
    main()
