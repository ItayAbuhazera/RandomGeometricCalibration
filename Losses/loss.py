from torch.nn import functional as F


def cross_entropy(logits, targets, **kwargs):
    return F.cross_entropy(logits, targets, reduction="sum")


def get_loss_function(loss_name, **kwargs):
    if loss_name not in LOSS_FUNCTIONS:
        available = list(LOSS_FUNCTIONS.keys())
        raise ValueError(f"Unknown loss function: {loss_name}. Available: {available}")

    loss_fn = LOSS_FUNCTIONS[loss_name]

    def loss_wrapper(logits, targets, **additional_kwargs):
        combined_kwargs = {**kwargs, **additional_kwargs}
        return loss_fn(logits, targets, **combined_kwargs)

    return loss_wrapper


LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy,
}


LOSS_CONFIGS = {
    "cross_entropy": {},
}


__all__ = ["cross_entropy", "get_loss_function", "LOSS_FUNCTIONS", "LOSS_CONFIGS"]
