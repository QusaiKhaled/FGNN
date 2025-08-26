import torch

from .preprocessor import SemiSupervisedPreprocessor


def get_data(parameters, logger):
    data_path = parameters["path"]
    window_size = parameters["window_size"]
    stride = parameters["stride"]
    train_ratio = parameters.get("train_ratio", 0.7)
    val_ratio = parameters.get("val_ratio", 0.1)
    max_windows = parameters.get("max_windows", None)
    anomaly_detection = parameters.get("anomaly_detection", False)
    
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
        val_ratio=val_ratio,
        max_windows=max_windows,
        anomaly_detection=anomaly_detection,
        logger=logger,
    )
    train_data, val_data, test_data = preprocessor.preprocess()
    logger.info(
        f"Training on {len(train_data)} windows; testing on {len(test_data)} windows."
    )

    return train_data, val_data, test_data