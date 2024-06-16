import click
import logging
logging.basicConfig(level=logging.INFO)

from nightwatch.data.readers import SleepAccelDataReader
from nightwatch.data.features import SleepFeatureExtractor
from nightwatch.data.datasets import SleepSeqDataset


@click.command()
@click.option(
    '--sleep-accel-dir',
    default="./data/sleep-accel",
    show_default=True,
    help="The directory containing the raw sleep "
         "acceleration data."
)
@click.option(
    '--target-dir',
    default="./data/sleep-accel-seq",
    show_default=True,
    help="The directory where the processed sleep "
         "acceleration sequence dataset will be saved."
)
@click.option(
    '--feat-window-size',
    default=50,
    show_default=True,
    help="The feature window size."
)
@click.option(
    '--min-seq-len',
    default=1,
    show_default=True,
    help="The minimum sequence length required to include "
         "a sample in the dataset."
)
@click.option(
    '--train-test-split',
    default=0.9,
    show_default=True,
    help="The ratio of training to testing data."
)
def build_sleep_accel_seq_ds(
        sleep_accel_dir: str = "../data/sleep-accel",
        target_dir: str = "../data/sleep-accel-seq",
        feat_window_size: int = 50,
        min_seq_len: int = 1,
        train_test_split: float = 0.9
):
    """Builds a sleep-accel dataset for sequence classification.

      Args:
        sleep_accel_dir (str, optional): The directory containing the raw
          sleep acceleration data. Defaults to '../data/sleep-accel'.
        target_dir (str, optional): The directory where the processed sleep
          acceleration sequence dataset will be saved. Defaults to '../data/sleep-accel-seq'.
    """
    reader = SleepAccelDataReader(sleep_accel_dir)
    extractor = SleepFeatureExtractor(window_size=feat_window_size)

    logging.info("Building dataset...")
    SleepSeqDataset.build_dataset(
        reader, extractor,
        target_dir=target_dir,
        min_seq_len=min_seq_len,
        train_test_split=train_test_split
    )
    logging.info("Build complete.")

                             
