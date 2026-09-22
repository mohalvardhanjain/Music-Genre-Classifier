import numpy as np
import librosa

from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm


SAMPLE_RATE = 44100
CHUNK_DURATION = 4
OVERLAP_DURATION = 2

N_MELS = 128
TARGET_SIZE = (150, 150)


def process_audio(file_path):
    """
    Convert one audio file into overlapping
    log-Mel spectrogram chunks.
    """

    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    chunk_samples = int(
        CHUNK_DURATION * SAMPLE_RATE
    )

    hop_samples = int(
        (CHUNK_DURATION - OVERLAP_DURATION)
        * SAMPLE_RATE
    )

    # Pad if audio is shorter than one chunk
    if len(y) < chunk_samples:

        y = np.pad(
            y,
            (0, chunk_samples - len(y))
        )

    chunks = []

    for start in range(
        0,
        len(y) - chunk_samples + 1,
        hop_samples
    ):

        chunk = y[
            start:start + chunk_samples
        ]

        mel = librosa.feature.melspectrogram(
            y=chunk,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=2048,
            hop_length=512
        )

        # Log compression
        mel = np.log(
            mel + 1e-9
        )

        # Resize to 150 × 150
        mel = zoom(
            mel,
            (
                TARGET_SIZE[0] / mel.shape[0],
                TARGET_SIZE[1] / mel.shape[1]
            ),
            order=1
        )

        chunks.append(
            mel.astype(np.float32)
        )

    return np.stack(chunks)


def process_dataset(
    input_dir,
    output_dir
):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    audio_files = sorted(
        input_dir.rglob("*.wav")
    )

    print(
        f"Found {len(audio_files)} audio files."
    )

    for file_path in tqdm(
        audio_files,
        desc="Processing audio"
    ):

        genre = file_path.parent.name

        genre_dir = output_dir / genre

        genre_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            genre_dir /
            f"{file_path.stem}.npy"
        )

        # Don't process again if already done
        if output_path.exists():
            continue

        try:

            chunks = process_audio(
                file_path
            )

            np.save(
                output_path,
                chunks
            )

        except Exception as e:

            print(
                f"\nSkipping {file_path}"
            )

            print(
                f"Reason: {e}"
            )


if __name__ == "__main__":
    
    INPUT_DIR = Path("data/split")
    OUTPUT_DIR = Path("data/processed")

    for split in ["train", "val", "test"]:

        print()
        print("=" * 50)
        print(f"Processing {split}")
        print("=" * 50)

        process_dataset(
            input_dir=INPUT_DIR / split,
            output_dir=OUTPUT_DIR / split
        )