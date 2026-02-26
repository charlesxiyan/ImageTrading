# ImageTrading

A CNN-based stock price trend prediction framework for Chinese A-share markets, inspired by the paper **(Re-)Imag(in)ing Price Trends** and the work at [RichardS0268/CNN-for-Trading](https://github.com/RichardS0268/CNN-for-Trading).

The core idea is to encode intraday OHLCV price data as 2D grayscale pixel images and train a convolutional neural network to classify future price direction (up/down).

---

## Overview

Traditional quantitative strategies rely on hand-crafted technical indicators. This project takes a different approach: raw OHLC price bars are drawn directly into pixel images, and a CNN learns visual patterns associated with future returns — without any feature engineering.

**Pipeline:**

```
Raw Intraday OHLCV Data
        │
        ▼
  Image Generation (imaging.py)
  [OHLC bars → 2D grayscale pixel images]
        │
        ▼
  CNN Training (training.py)
  [Binary classification: up / down]
        │
        ▼
  Inference & Factor Output (testing.py)
  [Model predictions → alpha factor cache]
```

---

## Models

Two model variants are provided in `cnn.py`, both sharing the architecture:
`Conv2d → BatchNorm → ReLU → MaxPool` × 2, followed by `Dropout → FC → Softmax`

| Model    | Lookback Window | Image Size   | Frequency |
|----------|-----------------|--------------|-----------|
| `CNN49u` | 49 bars         | 147 × 147 px | 5-minute  |
| `CNN9u`  | 9 bars          | 27 × 27 px   | 30-minute |

**Architecture details:**
- Conv Block 1: 1 → 64 channels, kernel (5×3), Xavier weight initialization
- Conv Block 2: 64 → 128 channels, kernel (5×3)
- Dropout: p = 0.5
- Fully connected: flattened features → 2 (binary classification)
- Loss: Binary Cross-Entropy (BCE)
- Optimizer: Adam with weight decay

---

## File Structure

```
ImageTrading/
├── __init__.py          # Global imports (PyTorch, NumPy, Pandas, etc.)
├── __rd__.py            # Supplementary imports
├── config.yaml          # Experiment configuration
├── yaml_create.py       # Helper to generate config files
├── initialsetting.py    # YAML → namedtuple config loader
├── imaging.py           # Data retrieval and pixel image generation
├── cnn.py               # CNN model definitions (CNN49u, CNN9u)
├── training.py          # Training loop with early stopping & LR scheduling
├── testing.py           # Inference and alpha factor cache generation
├── pj_cnn.py            # Project-level entry point / pipeline script
└── utils.py             # Utility functions (timer context manager)
```

---

## Configuration

All experiment parameters are managed via `config.yaml`:

```yaml
MODEL: CNN49u          # CNN49u or CNN9u

PATHS:
  COMMONCACHE_DIR: /path/to/commoncache      # Root dir of raw market data cache
  IMAGE_DATA_DIR:  /path/to/image/data       # Where generated pixel images are saved
  PROJECT_DIR:     /path/to/project/output   # Root for TensorBoard logs and test outcomes

DATASET:
  LOOKBACK_WIN: 49     # Number of bars per image (9 or 49)
  START_DATE: 20211101
  END_DATE: 20211110
  SAMPLE_RATE: 0.2     # Fraction of samples to generate (for data efficiency)
  SHOW_VOLUME: false
  PARALLEL_NUM: 12     # CPU cores for parallel image generation
  INDICATORS:
    MA:
      WIN: 245         # Moving average window (overlaid on image)

TRAIN:
  START_DATE: 20170101
  END_DATE: 20201231
  PREDICT_WIN: 50      # Prediction horizon (bars ahead)
  LABEL: RET_LONG      # RET_LONG or RET_SHORT
  NEPOCH: 20
  BATCH_SIZE: 256
  LEARNING_RATE: 0.05
  LR_BASE_RATE: 0.001
  WARMUP_EPOCH: 1
  WEIGHT_DECAY: 0.01
  VALID_RATIO: 0.06
  EARLY_STOP_EPOCH: 5
  MODEL_SAVE_FILE: /path/to/models/model.tar
  LOG_SAVE_FILE: /path/to/logs/log.csv

INFERENCE:
  START_DATE: 20210101
  END_DATE: 20230630
  FACTORS_SAVE_FILE: /path/to/factors/factors.csv
```

---

## Key Components

### Image Generation (`imaging.py`)

The `Imaging` class reads raw memory-mapped intraday OHLCV data and converts each stock's price history into a series of grayscale images:

- Each image covers `LOOKBACK_WIN` bars (e.g., 49 × 5-min bars ≈ 1 trading day)
- Open/Close are plotted as single pixels; the High–Low range fills the middle column of each 3-pixel-wide bar
- Price data is rescaled to `[0, image_size]`; limit-up/limit-down days are handled explicitly
- Images are saved as `.pkl` batches, organized by ticker
- Parallel generation via `joblib.Parallel` across all A-share stocks

### Training (`training.py`)

The `Training` class supports:

- **Intra-epoch validation**: validation runs at fixed checkpoints within each epoch (e.g., every 25% of batches), enabling early stopping without waiting for a full epoch
- **Early stopping**: stops training when validation loss/accuracy fails to improve for `EARLY_STOP_EPOCH` consecutive validations
- **LR scheduling**: Cosine Annealing, Step, or Exponential decay (with optional linear warmup)
- **TensorBoard logging**: training/validation loss and accuracy tracked in real time
- **Model checkpointing**: best model saved to disk whenever validation improves

### Inference (`testing.py`)

The `Testing` class loads a trained model and runs inference over the full stock universe:

- Outputs per-stock, per-timestep softmax logits (probability of upward movement)
- Results are stored in a memory-mapped cache array of shape `(trading_days, stocks, intervals)` for downstream factor research
- Supports time-shift alignment between prediction timestamp and holding period

---

## Dependencies

- Python 3.8+
- PyTorch
- NumPy / Pandas / Polars
- scikit-learn / imbalanced-learn
- joblib
- tqdm
- PyYAML
- psutil
- plotly / matplotlib

---

## Acknowledgements

This project is inspired by:

- Paper: **(Re-)Imag(in)ing Price Trends** — which demonstrates that CNNs applied to raw price images can learn meaningful predictive signals for stock returns
- Repository: [RichardS0268/CNN-for-Trading](https://github.com/RichardS0268/CNN-for-Trading)
