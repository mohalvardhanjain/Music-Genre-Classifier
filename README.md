# Music Genre Classification using CNN

A deep learning project for classifying music into 10 genres using **Convolutional Neural Networks (CNNs)** and **Mel-spectrograms** generated directly from raw audio.

## Genres

* Blues
* Classical
* Country
* Disco
* Hip-hop
* Jazz
* Metal
* Pop
* Reggae
* Rock

## Pipeline

```text
Raw Audio
    ↓
70/15/15 Track-Level Split
    ↓
4-sec Audio Segments
(2-sec overlap)
    ↓
Mel-Spectrogram
    ↓
Log Transformation
    ↓
150 × 150 Spectrogram
    ↓
CNN
    ↓
Genre Prediction
```

The dataset is split **before segmentation** to prevent audio segments from the same track appearing in different splits.

## Preprocessing

| Parameter        |     Value |
| ---------------- | --------: |
| Dataset          |     GTZAN |
| Sample Rate      |  44.1 kHz |
| Segment Duration |     4 sec |
| Overlap          |     2 sec |
| Mel Bands        |       128 |
| FFT Size         |      2048 |
| Hop Length       |       512 |
| Spectrogram Size | 150 × 150 |

Each spectrogram is transformed using:

$$
M_{log} = \log(M + 10^{-9})
$$

## CNN Architecture

```text
Conv2D(32) → Conv2D(32) → MaxPool
Conv2D(64) → Conv2D(64) → MaxPool
Conv2D(128) → Conv2D(128) → MaxPool → Dropout
Conv2D(256) → Conv2D(256) → MaxPool
Conv2D(512) → Conv2D(512) → MaxPool → Dropout
Flatten
Dense(1200) → Dropout
Dense(10) → Softmax
```

The model contains approximately **7.18 million parameters**.

### Training

* Optimizer: Adam
* Learning rate: `1e-4`
* Batch size: `32`
* Loss: Sparse Categorical Crossentropy

## Results

Two training strategies were compared:

| Training Strategy            | Track-Level Accuracy |
| ---------------------------- | -------------------: |
| One random segment per track |               62.67% |
| All available segments       |           **83.33%** |

The final model achieved:

* **76.62%** chunk-level test accuracy
* **83.33%** track-level test accuracy

For track-level prediction, all segments of a song are classified and their softmax probabilities are averaged to produce the final genre prediction.

### Per-Genre Accuracy

| Genre     | Accuracy |
| --------- | -------: |
| Blues     |   93.33% |
| Classical |  100.00% |
| Country   |   73.33% |
| Disco     |   80.00% |
| Hip-hop   |   86.67% |
| Jazz      |   73.33% |
| Metal     |   80.00% |
| Pop       |   93.33% |
| Reggae    |   73.33% |
| Rock      |   80.00% |

## Project Structure

```text
music-genre-classifier/
├── data/
│   ├── raw/
│   ├── split/
│   └── processed/
├── models/
│   └── cnn_all_chunks.keras
├── notebooks/
│   ├── 01_audio_exploration.ipynb
│   └── 02_cnn_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   └── model.py
├── requirements.txt
└── README.md
```

## Technologies

Python · TensorFlow/Keras · Librosa · NumPy · SciPy · Scikit-learn · Matplotlib · Jupyter

## Future Improvements

* Data augmentation and SpecAugment
* Batch normalization and learning-rate scheduling
* Transfer learning
* Comparison with MFCCs and other audio representations
* Real-time/web-based genre prediction

## Dataset

This project uses the **GTZAN Genre Collection** for experimentation and research purposes.
