
import random
import numpy as np
import torch




def set_seed(seed, strict=False):
    """
    Set random seeds for reproducible behavior in Python, NumPy, and PyTorch.

    Seeds the Python built-in RNG, NumPy RNG, and PyTorch RNG for both
    CPU and all available CUDA devices. When ``strict=True``, enables
    deterministic cuDNN behavior and enforces deterministic PyTorch
    algorithms, which may reduce performance.

    Parameters
    ----------
    seed : int
        Random seed value.
    strict : bool, default=False
        If True, enforce deterministic cuDNN behavior and PyTorch
        deterministic algorithms.

    Returns
    -------
    None
    """

    # set python, numpy, pytorch seeds...
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # optionally set pytorch deterministic settings (may slow training)...
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)





def set_worker_seed(worker_id):
    """
    Initialize random seeds for a PyTorch DataLoader worker.

    Derives a worker-specific seed from ``torch.initial_seed()`` and
    uses it to seed NumPy and Python's built-in random module. Intended
    for use as the ``worker_init_fn`` argument in ``torch.utils.data.DataLoader``
    to ensure reproducible data loading when ``num_workers > 0``.

    Parameters
    ----------
    worker_id : int
        Worker process identifier assigned by the DataLoader.

    Returns
    -------
    None
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
