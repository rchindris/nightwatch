import click
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


from nightwatch.data.datasets import SleepSeqDataset
from nightwatch.models.seq_cls import LSTMSeqClassifier

def collate_fn(batch):
    """
    Custom collate function to pad sequences with 0.

    Args:
        batch (list of tuples): A batch of (sequence, label) tuples.

    Returns:
        tuple: (padded_sequences, labels) where padded_sequences is a
          tensor of shape (batch_size, max_sequence_length, num_features)
               and labels is a tensor of shape (batch_size).
    """
    seq, labels = zip(*batch)

    seq_padded = pad_sequence(
        [torch.tensor(s, dtype=torch.float32) for s in seq],
        batch_first=True,
        padding_value=0)
    
    labels = torch.tensor(labels, dtype=torch.int64)
    lengths = torch.tensor([len(s) for s in seq])

    return seq_padded, lengths, labels


class SeqClassifierTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training an LSTM-based sequence classifier.

    Args:
        model (nn.Module): The LSTM sequence classifier model.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
    """

    def __init__(self, model, lr=1e-3):
        super(SeqClassifierTrainer, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x, lengths):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_features).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_labels).
        """
        return self.model(x, lengths)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (tuple): A batch of data (sequences, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The training loss.
        """
        sequences, lengths, labels = batch
        outputs = self(sequences, lengths)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (tuple): A batch of data (sequences, labels).
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The validation loss.
        """
        sequences, lengths, labels = batch
        outputs = self(sequences, lengths)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@click.command()
@click.option("--ds_dir", default="./data/sleep-accel-seq",
              help="Path to the dataset dir.")
@click.option("--model_path", default="./exps",
              help="Path for saving model checkpoints and training logs.")
@click.option('--num_units', default=128,
              help='Number of hidden units in the LSTM.')
@click.option('--num_layers', default=1,
              help='Number of LSTM layers.')
@click.option('--dropout', default=0.2,
              help='Dropout rate.')
@click.option('--lr', default=1e-3,
              help='Learning rate for the optimizer.')
@click.option('--max_epochs', default=20,
              help='Number of epochs to train the model.')
@click.option('--batch_size', default=32,
              help='Batch size for training.')
def train_sleep_accel(ds_dir, model_path, num_units,
                      num_layers, dropout, lr, max_epochs,
                      batch_size):
    """
    Train the LSTM sequence classifier on the sleep dataset.
    """
    ds_path = Path(ds_dir)
    exp_dir = Path(model_path)
    
    train_dataset = SleepSeqDataset(ds_path / "train.json")
    test_dataset = SleepSeqDataset(ds_path / "test.json")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=collate_fn,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             collate_fn=collate_fn)

    # Define the checkpoint directory
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = exp_dir / f"seq_cls/lstm_{num_units}_{num_layers}_sleep_accel/ckpt"

    # Callback for saving the best model separately
    best_model_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="seq_cls_{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    # Instantiate the LSTMSeqClassifier model
    lstm_model = LSTMSeqClassifier(train_dataset.num_features,
                                   num_units, train_dataset.num_labels,
                                   num_layers, dropout)

    # Instantiate the LSTMClassifier for training
    model = SeqClassifierTrainer(lstm_model, lr)

    # Set up trainer with callbacks
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[best_model_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    train_sleep_accel()
