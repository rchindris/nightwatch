# NightWatch - Sleep Stage Classifiers

Datasets and models for sleep stage prediction.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. Follow the steps below to set up the project.

1. **Install Poetry**:

   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install Dependencies**:

   ```sh
   poetry update && poetry install
   ```

## Download and prepare the Sleep-Accel dataset

The Sleep Accel dataset contains motion and heart rate measurements collected from 31 subjects using an Apple Watch, and it is available from [here](https://www.physionet.org/content/sleep-accel/1.0.0/).

Datasets are stored under `/data`. To download and prepare Sleep-Accel, use the `data/download.sh` script. By default, the script creates a `sleep-accel` directory in the current working directory, where the dataset samples are downloaded and prepared.

```sh
cd data/
./download.sh
```

For convenience, all other scripts assume that the dataset is under `./data/sleep-accel`. To override the target directory, use the `--data-dir` argument:

```sh
./data/download.sh --data-dir some/other/path
```

The script executes three stages: download, prepare, and cleanup. To execute each stage independently, run the script for the specific stage:

```sh
cd data/
./download.sh cleanup
```

## Extract Features

The dataset contains polysomnography (PSG) annotated samples, sampled every 30 seconds. To compute features for each PSG sample and prepare sample sequences for sequence classification, use the `build_seq_ds` tool:

```sh
poetry run build_seq_ds
```

By default, this tool assumes the Sleep-Accel dataset resides under `./data/sleep-accel` and creates a new directory under `./data/sleep-accel-seq` containing sequences for each user with a specific time length, in minutes. By default, 10-minute length sequences are extracted with a stride of 5 minutes. To change the window size or stride, e.g., to 1 hour sampled every minute, do:

```sh
poetry run build_seq_ds --window-size-min 60 --window-stride-min 1
```

For convenience, the data loaders assume the sequence dataset resides under `./data/sleep-accel-seq`. To change the source and target directories, use the `--sleep-accel-dir` and `--target-dir` arguments:

```sh
poetry run build_seq_ds --window-size-min 60 --window-stride-min 1 --sleep-accel-dir raw/sleep/accel --target-dir ./data/sleep-accel-1h-every-1min
```

Execute `poetry run build_seq_ds --help` to check usage documentation. The supported parameters are:

- `--sleep-accel-dir`: The directory containing the raw sleep acceleration data. (Default: `./data/sleep-accel`)
- `--target-dir`: The directory where the processed sleep acceleration sequence dataset will be saved. (Default: `./data/sleep-accel-seq`)
- `--window-size-min`: The input window size in minutes. (Default: 10)
- `--window-stride-min`: The input stride in minutes. (Default: 5)
- `--train-test-split`: The ratio of training to testing data. (Default: 0.9)

## Performing Experiments

To train sequence classifiers on the PSG-annotated data, use the `train_seq_cls` tool:

```sh
poetry run train_seq_cls --batch_size 16 --max_epochs 100
```

This will use the data from `./data/sleep-accel-seq` to train a default 1-layer LSTM sequence encoder followed by an MLP classifier. To experiment with different model configurations, pass in the hyperparameters to the `train_seq_cls` tool. Supported parameters are:

- `--ds_dir`: Path to the dataset directory. (Default: `./data/sleep-accel-seq`)
- `--model_path`: Path for saving model checkpoints and training logs. (Default: `./exps`)
- `--num_units`: Number of hidden units in the LSTM. (Default: 128)
- `--num_layers`: Number of LSTM layers. (Default: 1)
- `--dropout`: Dropout rate. (Default: 0.2)
- `--lr`: Learning rate for the optimizer. (Default: 1e-3)
- `--max_epochs`: Number of epochs to train the model. (Default: 20)
- `--batch_size`: Batch size for training. (Default: 32)

## Results

| Window Size | Window Stride | Num Layers | Units | ACC  |
|-------------|---------------|------------|-------|------|
| 60          | 1             | 2          | 32    | 81.4 |
| 60          | 1             | 4          | 32    | 83.7 |
