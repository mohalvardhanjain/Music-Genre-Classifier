from pathlib import Path
import random
import shutil


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

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def split_dataset(
    input_dir="data/raw/genres_original",
    output_dir="data/split"
):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    random.seed(SEED)

    for genre in GENRES:

        genre_dir = input_dir / genre

        files = sorted(
            genre_dir.glob("*.wav")
        )

        random.shuffle(files)

        total = len(files)

        train_end = int(
            TRAIN_RATIO * total
        )

        val_end = train_end + int(
            VAL_RATIO * total
        )

        train_files = files[:train_end]

        val_files = files[
            train_end:val_end
        ]

        test_files = files[
            val_end:
        ]

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files
        }

        for split, split_files in splits.items():

            split_genre_dir = (
                output_dir /
                split /
                genre
            )

            split_genre_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for file_path in split_files:

                destination = (
                    split_genre_dir /
                    file_path.name
                )

                shutil.copy2(
                    file_path,
                    destination
                )

        print(
            f"{genre}: "
            f"train={len(train_files)}, "
            f"val={len(val_files)}, "
            f"test={len(test_files)}"
        )


if __name__ == "__main__":

    split_dataset()