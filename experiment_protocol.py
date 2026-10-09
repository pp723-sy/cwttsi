import numpy as np


def temporal_split_indices(num_samples, history_step, target_step, protocol="window_ratio"):
    """Return chronological train/validation/test indices.

    target_time keeps target timestamps disjoint across adjacent splits. Historical
    inputs may reach into the preceding split because those values are observable
    at prediction time.
    """
    if protocol == "window_ratio":
        train_size = int(num_samples * 0.6)
        val_size = int(num_samples * 0.2)
        indices = np.arange(num_samples, dtype=np.int64)
        return (
            indices[:train_size],
            indices[train_size:train_size + val_size],
            indices[train_size + val_size:],
        )
    if protocol != "target_time":
        raise ValueError("protocol must be 'window_ratio' or 'target_time'.")

    total_steps = num_samples + history_step + target_step - 1
    train_boundary = int(total_steps * 0.6)
    val_boundary = int(total_steps * 0.8)

    def target_bounded(start, end):
        first = max(0, start - history_step)
        last = min(num_samples - 1, end - history_step - target_step)
        if last < first:
            return np.empty(0, dtype=np.int64)
        return np.arange(first, last + 1, dtype=np.int64)

    train_idx = target_bounded(history_step, train_boundary)
    val_idx = target_bounded(train_boundary, val_boundary)
    test_idx = target_bounded(val_boundary, total_steps)
    if min(map(len, (train_idx, val_idx, test_idx))) == 0:
        raise ValueError("Not enough samples for target-disjoint 6:2:2 splitting.")
    return train_idx, val_idx, test_idx

