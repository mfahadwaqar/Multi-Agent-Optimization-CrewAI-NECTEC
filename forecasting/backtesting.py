"""Chronological holdout and expanding-window fold generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train: slice
    validation: slice


def chronological_holdout(length: int, test_fraction: float, gap: int) -> tuple[slice, slice]:
    test_start = int(length * (1.0 - test_fraction))
    train_end = test_start - gap
    if train_end < 10 or test_start >= length:
        raise ValueError("Not enough time-series rows for the requested test split and horizon gap")
    return slice(0, train_end), slice(test_start, length)


def expanding_window_folds(length: int, folds: int, validation_size: int, gap: int, minimum_train_size: int) -> list[Fold]:
    folds = int(folds)
    validation_size = int(validation_size)
    first_validation = length - folds * validation_size
    if first_validation - gap < minimum_train_size:
        raise ValueError("Not enough development rows for expanding-window validation")
    result = []
    for number in range(folds):
        validation_start = first_validation + number * validation_size
        result.append(Fold(slice(0, validation_start - gap), slice(validation_start, validation_start + validation_size)))
    return result
