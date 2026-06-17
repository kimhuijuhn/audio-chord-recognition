import numpy as np
import mir_eval
from audio_chord_recognition.features import get_frame_times

# template
C_MAJOR_INTERVAL = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
C_MINOR_INTERVAL = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
TEMPLATES = []
for i in range(12):
    TEMPLATES.append(np.roll(C_MAJOR_INTERVAL, i))
for i in range(12):
    TEMPLATES.append(np.roll(C_MINOR_INTERVAL, i))
TEMPLATES = np.array(TEMPLATES)

# interval - name map
TEMPLATE_NAMES = [
    "C:maj",
    "C#:maj",
    "D:maj",
    "D#:maj",
    "E:maj",
    "F:maj",
    "F#:maj",
    "G:maj",
    "G#:maj",
    "A:maj",
    "A#:maj",
    "B:maj",
    "C:min",
    "C#:min",
    "D:min",
    "D#:min",
    "E:min",
    "F:min",
    "F#:min",
    "G:min",
    "G#:min",
    "A:min",
    "A#:min",
    "B:min",
]


def predict(chroma: np.ndarray) -> np.ndarray:
    """
    Given a chroma vector of shape (12, T), return a T-length array of chord
    strings using cosine similarity.
    """
    # verify chroma shape
    if chroma.shape[0] != 12:
        raise ValueError("chroma does not follow shape (12, T)")

    # calculate cosine similarity
    dot_product = TEMPLATES @ chroma  # shape: (24, T)
    template_norm = np.linalg.norm(TEMPLATES, axis=1)
    frame_norm = np.linalg.norm(chroma, axis=0)
    cosine = dot_product / template_norm[:, None] / frame_norm
    indices = np.argmax(cosine, axis=0)

    # return (interval, label)
    return np.array([TEMPLATE_NAMES[i] for i in indices])


def add_interval(
    prediction: np.ndarray, timestamps: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    est_intervals = []
    est_labels = []
    v_prev = prediction[0]  # first chord
    i_prev = 0

    # loop through prediction frame-wise
    for i, v in enumerate(prediction):
        if v_prev != v:
            est_intervals.append([timestamps[i_prev], timestamps[i]])
            est_labels.append(v_prev)
            v_prev = v
            i_prev = i

    # append last frame data
    est_intervals.append([timestamps[i_prev], timestamps[-1]])
    est_labels.append(v_prev)

    est_intervals = np.array(est_intervals)
    return est_intervals, est_labels


def evaluate(
    ref_intervals: np.ndarray, ref_labels: list, chroma: np.ndarray
) -> np.float64:
    """
    Derive a evaluation score using mir_eval.chord.majmin.
    Specifically in GuitarSet, hdim7 chords are not evaluated.
    """
    # build estimated intervals and labels
    prediction = predict(chroma)
    timestamps = get_frame_times(chroma)
    est_intervals, est_labels = add_interval(prediction, timestamps)

    # adjust estimated intervals to span the same time range as reference
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals,
        est_labels,
        ref_intervals.min(),
        ref_intervals.max(),
        mir_eval.chord.NO_CHORD,
        mir_eval.chord.NO_CHORD,
    )

    # merge ref and est to a common set of intervals
    intervals, merged_ref_labels, est_labels = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels
    )

    # durations of each merged interval
    durations = mir_eval.util.intervals_to_durations(intervals)

    # compare under maj-min rules
    comparisons = mir_eval.chord.majmin(merged_ref_labels, est_labels)

    # return score
    return mir_eval.chord.weighted_accuracy(comparisons, durations)
