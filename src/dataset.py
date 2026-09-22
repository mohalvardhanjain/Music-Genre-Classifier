from pathlib import Path

import numpy as np
import tensorflow as tf


GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]


GENRE_TO_LABEL = {
    genre: i
    for i, genre in enumerate(GENRES)
}


PROCESSED_DIR = Path(
    r"C:\Users\Mohal\Documents\Projects\music-genre-classifier\data\processed"
)


def get_processed_files(split="train"):
    split_dir = PROCESSED_DIR / split

    files = []

    for genre in GENRES:
        genre_dir = split_dir / genre

        for file in sorted(genre_dir.glob("*.npy")):
            files.append(
                (
                    str(file),
                    GENRE_TO_LABEL[genre]
                )
            )

    return files


def data_generator(
    files,
    training=True
):

    for filepath, label in files:

        spectrograms = np.load(filepath)

        if training:

            # Randomly select ONE chunk
            index = np.random.randint(
                0,
                len(spectrograms)
            )

        else:

            # Always use the same chunk
            index = 0

        spectrogram = spectrograms[index]

        # Add channel dimension
        spectrogram = spectrogram[..., np.newaxis]

        yield (
            spectrogram.astype(np.float32),
            np.int32(label)
        )


def create_dataset(
    split="train",
    batch_size=32,
    shuffle=False,
    repeat=False
):

    files = get_processed_files(
        split=split
    )

    training = split == "train"

    dataset = tf.data.Dataset.from_generator(

        lambda: data_generator(
            files,
            training=training
        ),

        output_signature=(

            tf.TensorSpec(
                shape=(150, 150, 1),
                dtype=tf.float32
            ),

            tf.TensorSpec(
                shape=(),
                dtype=tf.int32
            )
        )
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=len(files),
            reshuffle_each_iteration=True
        )

    if repeat:

        dataset = dataset.repeat()

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset