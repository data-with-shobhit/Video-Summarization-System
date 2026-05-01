"""Download MoViNet-A2-Stream weights from TensorFlow Hub."""
from __future__ import annotations

import argparse
import os

import tensorflow as tf
import tensorflow_hub as hub


def download(model: str, output_dir: str) -> None:
    model_map = {
        "a0": "https://tfhub.dev/tensorflow/movinet/a0/stream/kinetics-600/classification/3",
        "a2": "https://tfhub.dev/tensorflow/movinet/a2/stream/kinetics-600/classification/3",
    }
    if model not in model_map:
        raise ValueError(f"Unknown model: {model}. Choose from {list(model_map)}")

    url = model_map[model]
    print(f"Downloading MoViNet-{model.upper()}-Stream from TF Hub...")
    loaded = hub.load(url)
    tf.saved_model.save(loaded, output_dir)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="a2", choices=["a0", "a2"])
    parser.add_argument("--output", default="./models/movinet_a2_stream")
    args = parser.parse_args()
    download(args.model, args.output)
