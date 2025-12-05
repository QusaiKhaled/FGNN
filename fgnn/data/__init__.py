import torch

from .preprocessor import SemiSupervisedPreprocessor, add_drift


def get_data(parameters, logger):
    data_path = parameters["path"]
    window_size = parameters["window_size"]
    stride = parameters["stride"]
    train_ratio = parameters.get("train_ratio", 0.7)
    val_ratio = parameters.get("val_ratio", 0.1)
    max_windows = parameters.get("max_windows", None)
    anomaly_detection = parameters.get("anomaly_detection", False)
    split = parameters.get("split", None)
    
    logger.info(
        f"Loading data from {data_path} with window size {window_size}, stride {stride}, "
        f"train ratio {train_ratio}, and max windows {max_windows}."
    )

    raw_data = torch.load(data_path, map_location="cpu", weights_only=False)
    
    if split == "year":
        logger.info("Using year-based split")
        trainval_data_len = sum(raw_data.year_len[:-1]).item()
        test_data_len = raw_data.x.shape[1] - trainval_data_len
        train_data_len = int(trainval_data_len * train_ratio)
        val_data_len = trainval_data_len - train_data_len
        train_ratio = train_data_len / raw_data.x.shape[1]
        val_ratio = val_data_len / raw_data.x.shape[1]
        test_ratio = test_data_len / raw_data.x.shape[1]
        logger.info(f"Calculated train ratio: {train_ratio}, val ratio: {val_ratio}, test ratio: {test_ratio}")
    
    logger.info(
        f"Raw data loaded with {len(raw_data)} samples. Preprocessing to create windows."
    )
    drift_parameters = parameters.get("drift", None)
    if drift_parameters is not None:
        logger.info("Applying drift to the data as per drift parameters.")
        raw_data = add_drift(
            raw_data,
            **drift_parameters
        )
        logger.info("Drift applied.")
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