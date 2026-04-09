
import os
import torchinfo



def architecture_to_json(output_dir, model, loader):
    """
    Generate and save a model architecture summary to a JSON file.

    Parameters
    ----------
    output_dir : str or os.PathLike
        Directory where the architecture file will be written.
    model : torch.nn.Module
        Model to summarize.
    loader : iterable
        Data loader providing input batches. The first batch is used to
        infer the input shape. If present, keys "label" or "mask" are
        excluded from inputs.

    Returns
    -------
    None
    """

    batch = next(iter(loader))
    
    if "label" in batch.keys():
        x = {k: v for k, v in batch.items() if k != "label"}
    
    elif "mask" in batch.keys():
        x = {k: v for k, v in batch.items() if k != "mask"}

    x = next(iter(x.values()))

    input_size = tuple(x[:1].shape)

    architecture = torchinfo.summary(model, input_size=input_size, depth=4, verbose=0, col_names=["input_size", "kernel_size", "output_size", "num_params"])

    output_path = os.path.join(output_dir, "architecture.json")
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(str(architecture))