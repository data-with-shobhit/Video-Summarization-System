from __future__ import annotations

import os
import numpy as np

_model = None


def load_model():
    """Load MoViNet-A2-Stream. Pre-warm to avoid cold-start on first video."""
    global _model
    if _model is not None:
        return _model

    import tensorflow as tf  # lazy import — only needed when visual lane runs

    model_path = os.environ.get("MOVINET_MODEL_PATH", "./models/movinet_a2_stream")
    _model = tf.saved_model.load(model_path)

    dummy = tf.zeros([1, 1, 172, 172, 3])
    init_states = _model.init_states(tf.constant([1, 1, 172, 172, 3], dtype=tf.int32))
    _model({"image": dummy, **init_states})  # pre-warm — return value ignored

    return _model


def embed_frames(frames: list, model=None) -> np.ndarray:
    """
    frames: list of (172, 172, 3) float32 numpy arrays
    Returns: (N_frames, 600) numpy array
    """
    import tensorflow as tf

    if model is None:
        model = load_model()

    # init_states needs full 5D shape: [batch, frames, H, W, C]
    init_shape = tf.constant([1, 1, 172, 172, 3], dtype=tf.int32)
    states = model.init_states(init_shape)
    embeddings = []

    for frame in frames:
        # frame is (172, 172, 3) → expand to (1, 1, 172, 172, 3)
        frame_t = tf.cast(frame, tf.float32)[tf.newaxis, tf.newaxis]
        outputs, states = model({"image": frame_t, **states})
        # outputs is the logits tensor directly, not a dict
        logits = outputs.numpy() if hasattr(outputs, "numpy") else outputs["logits"].numpy()
        embeddings.append(logits[0])

    return np.array(embeddings)  # (N_frames, 600)


def compute_visual_change_rate(embeddings: np.ndarray) -> float:
    """Mean frame-to-frame L2 distance — proxy for visual activity."""
    if len(embeddings) < 2:
        return 0.0
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    # normalize by embedding magnitude so score is 0-1 range
    mean_mag = np.mean(np.linalg.norm(embeddings, axis=1)) + 1e-8
    return float(np.mean(diffs) / mean_mag)
