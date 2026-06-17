import pytest
import numpy as np
import librosa

from audio_chord_recognition import features
from audio_chord_recognition.hmm import template_matching as tm


@pytest.fixture
def tones():
    freqs = [261.63, 293.66, 329.63, 349.23, 392, 440, 493.88]
    tones = []
    for f in freqs:
        tones.append(librosa.tone(f, duration=1, sr=22050))

    return tones


@pytest.fixture
def chords(tones):
    c_major = tones[0] + tones[2] + tones[4]
    g_major = tones[1] + tones[4] + tones[6]
    a_minor = tones[0] + tones[2] + tones[5]
    f_major = tones[0] + tones[3] + tones[5]
    chords = [c_major, g_major, a_minor, f_major]
    return chords


@pytest.fixture
def chroma_cmaj(chords):
    return features.extract_chroma(chords[0])


@pytest.fixture
def chroma_amin(chords):
    return features.extract_chroma(chords[2])


# Testing predict()


def test_predict_raise_exception():
    with pytest.raises(ValueError):
        tm.predict(np.arange(24).reshape(6, 4))


def test_predict_major_chord(chroma_cmaj):
    predict = tm.predict(chroma_cmaj)
    assert np.all(predict == "C:maj")


def test_predict_minor_chord(chroma_amin):
    predict = tm.predict(chroma_amin)
    assert np.all(predict == "A:min")


# Testing add_interval()


@pytest.fixture
def predictions_cmaj(chroma_cmaj):
    return tm.predict(chroma_cmaj)


@pytest.fixture
def timestamps_cmaj(chroma_cmaj):
    return features.get_frame_times(chroma_cmaj)


def test_add_interval_start_frame(predictions_cmaj, timestamps_cmaj):
    intervals, _ = tm.add_interval(predictions_cmaj, timestamps_cmaj)
    assert intervals[0][0] == 0


def test_add_interval_end_frame(predictions_cmaj, timestamps_cmaj):
    intervals, _ = tm.add_interval(predictions_cmaj, timestamps_cmaj)
    assert intervals[0][1] == timestamps_cmaj[-1]


def test_add_interval_labels(predictions_cmaj, timestamps_cmaj):
    _, labels = tm.add_interval(predictions_cmaj, timestamps_cmaj)
    assert np.all(labels == ["C:maj"])


# Testing evaluate()


def test_evaluate_right(chroma_cmaj):
    ref_intervals = np.array([[0, 1]])
    ref_labels = ["C:maj"]
    assert np.isclose(tm.evaluate(ref_intervals, ref_labels, chroma_cmaj), 1, atol=0.1)


def test_evaluate_wrong(chroma_cmaj):
    ref_intervals = np.array([[0, 1]])
    ref_labels = ["F:maj"]
    assert np.isclose(tm.evaluate(ref_intervals, ref_labels, chroma_cmaj), 0, atol=0.1)
