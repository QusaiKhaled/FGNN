import torch

from .preprocessor import SemiSupervisedPreprocessor


def get_data(parameters, logger):
    data_path = parameters["path"]
    window_size = parameters.get("window_size", 24)
    stride = parameters.get("stride", 6)
    train_ratio = parameters.get("train_ratio", 0.8)
    max_windows = parameters.get("max_windows", 500)
    
    logger.info(
        f"Loading data from {data_path} with window size {window_size}, stride {stride}, "
        f"train ratio {train_ratio}, and max windows {max_windows}."
    )

    raw_data = torch.load(data_path, map_location="cpu", weights_only=False)
    
    logger.info(
        f"Raw data loaded with {len(raw_data)} samples. Preprocessing to create windows."
    )
    preprocessor = SemiSupervisedPreprocessor(
        raw_data,
        window_size=window_size,
        stride=stride,
        train_ratio=train_ratio,
        max_windows=max_windows,
    )
    train_data, test_data = preprocessor.preprocess()
    logger.info(
        f"Training on {len(train_data)} windows; testing on {len(test_data)} windows."
    )

    return train_data, test_data