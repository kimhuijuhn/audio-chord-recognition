import pytest
import numpy as np
import librosa
import soundfile as sf

from audio_chord_recognition import features


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
def money_chord(chords):
    # C-G-Am-F progressions
    money_chord = np.concatenate(chords)
    return money_chord


def test_extract_chroma_from_file_shape(tmp_path, money_chord):
    # store to file
    sf.write(tmp_path / "money_chord.wav", money_chord, 22050)
    chroma = features.extract_chroma_from_file(tmp_path / "money_chord.wav")
    assert chroma.shape[0] == 12


def test_extract_chroma_from_file_resampling(tmp_path):
    expected_sr = 22050
    expected_hop = 2048
    test_tone = librosa.tone(440, duration=1, sr=44100)
    sf.write(tmp_path / "test.wav", test_tone, 44100)
    chroma = features.extract_chroma_from_file(tmp_path / "test.wav")
    assert chroma.shape[1] == round(expected_sr / expected_hop)


def test_extract_chroma_shape(money_chord):
    chroma = features.extract_chroma(money_chord)
    assert chroma.shape[0] == 12


def test_get_frame_times_duration(money_chord):
    chroma = features.extract_chroma(money_chord)
    assert len(features.get_frame_times(chroma)) == chroma.shape[1]


def test_extract_chroma_minmax_values(money_chord):
    chroma = features.extract_chroma(money_chord)
    assert (chroma.min() >= 0.0) and (chroma.max() <= 1.0)


def test_extract_chroma_bin_match(chords):
    # import C major
    chroma = features.extract_chroma(chords[0])
    magnitudes = chroma.sum(axis=1)
    top_three_indices = set(np.argpartition(magnitudes, -3)[-3:])
    assert top_three_indices == {0, 4, 7}  # C=0, E=4, G=7
