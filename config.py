from pathlib import Path


def get_config():
    """
    Returns the configuration dictionary for the model.
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "vi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "datasource": "vi_en-translation",
    }


def get_weights_file_path(config, epoch: str):
    """
    Returns the file path for the model weights for a specific epoch.

    Args:
        config (dict): The configuration dictionary.
        epoch (str): The epoch for which to get the weights file.

    Returns:
        str: The file path for the weights.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    """
    Finds the latest weights file in the weights folder.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        str or None: The path to the latest weights file, or None if no files exist.
    """
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))

    if not weights_files:
        return None

    weights_files.sort()
    return str(weights_files[-1])
