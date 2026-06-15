import os
import numpy as np
import librosa


def extract_chroma_from_file(filepath: str | os.PathLike) -> np.ndarray:
    """
    Extract chroma features from a given filepath. Audio is resampled to
    22050Hz, following librosa convention.
    """
    # call librosa.load with the given filename
    y, sr = librosa.load(filepath, sr=22050)

    # hand off to extract_chroma
    return extract_chroma(y, sr)


def extract_chroma(
    y: np.ndarray,
    sr: int = 22050,
    bins_per_octave: int = 36,
    n_octaves: int = 6,
    hop_length: int = 2048,
) -> np.ndarray:
    """
    Extract chroma features from a given audio signal in Numpy ndarray.
    Per-frame normalization applied by using librosa's default value of the
    `norm` parameter.
    Returns:
        ndarray of shape (12, T), following librosa convention.
    """
    return librosa.feature.chroma_cqt(
        y=y,
        sr=sr,
        bins_per_octave=bins_per_octave,
        n_octaves=n_octaves,
        hop_length=hop_length,
    )


def get_frame_times(
    chroma: np.ndarray, sr: int = 22050, hop_length: int = 2048
) -> np.ndarray:
    """
    Get ndarray of timestamps given an ndarray of chroma features.
    """
    frames = np.arange(chroma.shape[1])
    return librosa.frames_to_time(frames=frames, sr=sr, hop_length=hop_length)
