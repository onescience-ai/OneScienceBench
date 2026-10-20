import torch


CHANNELS = ("Ux", "Uy", "p")


def _validate_prediction_target(predictions, targets):
    if predictions.shape != targets.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {predictions.shape} != {targets.shape}"
        )
    if predictions.ndim != 4:
        raise ValueError(
            "predictions and targets must have shape (B, C, H, W), "
            f"got ndim={predictions.ndim}"
        )
    if predictions.shape[1] != len(CHANNELS):
        raise ValueError(
            f"expected {len(CHANNELS)} output channels, got {predictions.shape[1]}"
        )
    if predictions.shape[0] == 0 or predictions.shape[2] == 0 or predictions.shape[3] == 0:
        raise ValueError("batch and grid dimensions must be non-empty")


def compute_sample_normalized_grid_mse(predictions, targets):
    """Return differentiable per-channel MSE and their sum.

    Squared error is first averaged over each sample's HxW grid, then averaged
    over the batch. Total_MSE is the sum of Ux_MSE, Uy_MSE and p_MSE. This makes
    the objective independent of batch size and grid-point count.
    """
    _validate_prediction_target(predictions, targets)
    per_sample_channel_mse = (predictions - targets).square().mean(dim=(2, 3))
    channel_mse = per_sample_channel_mse.mean(dim=0)
    metrics = {
        f"{name}_MSE": channel_mse[index]
        for index, name in enumerate(CHANNELS)
    }
    metrics["Total_MSE"] = channel_mse.sum()
    return metrics


class GridMSEAccumulator:
    """Accumulate sample-normalized grid MSE without batch-size bias."""

    def __init__(self, device):
        self.channel_sum = torch.zeros(
            len(CHANNELS), dtype=torch.float64, device=device
        )
        self.sample_count = 0

    def update(self, predictions, targets):
        batch_metrics = compute_sample_normalized_grid_mse(predictions, targets)
        self.update_from_metrics(batch_metrics, int(predictions.shape[0]))

    def update_from_metrics(self, batch_metrics, batch_size):
        """Accumulate a batch result returned by the canonical metric helper."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        channel_mse = torch.stack(
            [batch_metrics[f"{name}_MSE"] for name in CHANNELS]
        ).to(device=self.channel_sum.device, dtype=torch.float64)
        self.channel_sum += channel_mse * batch_size
        self.sample_count += batch_size

    def compute(self):
        if self.sample_count == 0:
            raise RuntimeError("metric accumulator is empty")
        channel_mse = self.channel_sum / self.sample_count
        result = {
            f"{name}_MSE": float(channel_mse[index].cpu())
            for index, name in enumerate(CHANNELS)
        }
        result["Total_MSE"] = float(channel_mse.sum().cpu())
        result["num_samples"] = self.sample_count
        return result
