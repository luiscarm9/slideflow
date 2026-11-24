import torch
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Union, Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn import __version__ as sklearn_version
from packaging import version
from fastai.vision.all import (
    DataLoaders, Learner, SaveModelCallback, CSVLogger
)

from slideflow import log
from slideflow.model import torch_utils
from .._params import TrainerConfig

# -----------------------------------------------------------------------------

def train(learner, config, callbacks=None):
    """Train an attention-based multi-instance learning model with FastAI.

    Args:
        learner (``fastai.learner.Learner``): FastAI learner.
        config (``TrainerConfig``): Trainer and model configuration.

    Keyword args:
        callbacks (list(fastai.Callback)): FastAI callbacks. Defaults to None.
    """
    cbs = [
        SaveModelCallback(fname=f"best_valid", monitor=config.save_monitor),
        CSVLogger(),
    ]
    if callbacks:
        cbs += callbacks
    if config.fit_one_cycle:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit_one_cycle(n_epoch=config.epochs, lr_max=lr, cbs=cbs)
    else:
        if config.lr is None:
            lr = learner.lr_find().valley
            log.info(f"Using auto-detected learning rate: {lr}")
        else:
            lr = config.lr
        learner.fit(n_epoch=config.epochs, lr=lr, wd=config.wd, cbs=cbs)
    return learner

class OrdinalClassEncoder:
    """Encode categorical features as cumulative ordinal threshold vectors.

    For k sorted classes, creates k-1 binary bits:
    - Class 0: [0, 0, ..., 0]
    - Class 1: [0, ..., 0, 1]
    - ...
    - Class k-1: [1, 1, ..., 1]
    """

    def __init__(self):
        self.categories_ = None
        self.ordinal_map_ = None

    @staticmethod
    def _validate_1d_column(X) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[1] != 1:
            raise ValueError("X should have shape (n_samples, 1)")
        return arr[:, 0]

    def fit(self, X):
        col = self._validate_1d_column(X)
        cats = np.unique(col)
        if len(cats) < 2:
            raise ValueError("Ordinal classification requires at least 2 unique classes.")
        self.categories_ = [cats]
        num_bits = len(cats) - 1
        self.ordinal_map_ = {
            cat: [1.0 if j >= (num_bits - i) else 0.0 for j in range(num_bits)]
            for i, cat in enumerate(cats)
        }
        return self

    def transform(self, X) -> np.ndarray:
        col = self._validate_1d_column(X)
        return np.asarray([self.ordinal_map_[x] for x in col], dtype=np.float32)


# -----------------------------------------------------------------------------

def build_learner(
    config: TrainerConfig,
    bags: List[str],
    targets: npt.NDArray,
    train_idx: npt.NDArray[np.int_],
    val_idx: npt.NDArray[np.int_],
    unique_categories: npt.NDArray,
    outdir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    **dl_kwargs
) -> Tuple[Learner, Tuple[int, int]]:
    """Build a FastAI learner for training an MIL model.

    Args:
        config (``TrainerConfig``): Trainer and model configuration.
        bags (list(str)): Path to .pt files (bags) with features, one per patient.
        targets (np.ndarray): Category labels for each patient, in the same
            order as ``bags``.
        train_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the training set.
        val_idx (np.ndarray, int): Indices of bags/targets that constitutes
            the validation set.
        unique_categories (np.ndarray(str)): Array of all unique categories
            in the targets. Used for one-hot encoding.
        outdir (str): Location in which to save training history and best model.
        device (torch.device or str): PyTorch device.

    Returns:
        fastai.learner.Learner, (int, int): FastAI learner and a tuple of the
            number of input features and output classes.

    """
    log.debug("Building FastAI learner")

    # Prepare device.
    device = torch_utils.get_device(device)

    # Prepare data.
    # Set oh_kw to a dictionary of keyword arguments for OneHotEncoder,
    # using the argument sparse=False if the sklearn version is <1.2
    # and sparse_output=False if the sklearn version is >=1.2.
    if version.parse(sklearn_version) < version.parse("1.2"):
        oh_kw = {"sparse": False}
    else:
        oh_kw = {"sparse_output": False}

    if config.model_type == 'classification':
        encoder = OneHotEncoder(**oh_kw).fit(unique_categories.reshape(-1, 1))
    elif config.model_type == 'ordinal':
        encoder = OrdinalClassEncoder().fit(unique_categories.reshape(-1, 1))
    else:
        encoder = None

    # Build the dataloaders.
    train_dl = config.build_train_dataloader(
        bags[train_idx],
        targets[train_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            num_workers=1,
            device=device,
            pin_memory=True,
            **dl_kwargs
        )
    )
    val_dl = config.build_val_dataloader(
        bags[val_idx],
        targets[val_idx],
        encoder=encoder,
        dataloader_kwargs=dict(
            shufle=False,
            num_workers=8,
            persistent_workers=True,
            device=device,
            pin_memory=False,
            **dl_kwargs
        )
    )

    # Prepare model.
    batch = train_dl.one_batch()
    n_in, n_out = config.inspect_batch(batch)
    n_out = 1 if config.model_type == 'survival' else n_out
    model = config.build_model(n_in, n_out).to(device)

    if hasattr(model, 'relocate'):
        model.relocate()

    # Loss should weigh inversely to class occurences.
    if config.model_type in ['classification', 'multimodal'] and config.weighted_loss:
        counts = pd.Series(np.asarray(targets[train_idx]).ravel()).value_counts()
        weights = counts.sum() / counts
        weights /= weights.sum()
        # Fall back to weight 1.0 for classes that appear in encoder.categories_
        # but have zero samples in train_idx (e.g., a stratified split that
        # misses a rare class). Without this, weights.get returns None and the
        # subsequent torch.tensor(...) raises TypeError on a None entry.
        weights = torch.tensor(
            [weights.get(c, 1.0) for c in encoder.categories_[0]],
            dtype=torch.float32
        ).to(device)
        loss_kw = {"weight": weights}
    else:
        loss_kw = {}
    loss_func = config.loss_fn(**loss_kw)

    # Create learning and fit.
    dls = DataLoaders(train_dl, val_dl)
    learner = Learner(dls, model, loss_func=loss_func, metrics=config.get_metrics(), path=outdir)

    return learner, (n_in, n_out)
