from dataclasses import dataclass

@dataclass
class FeatureExtractorConfig:
    batch_size: int = 256
    num_workers: int = 8
    feature_size: int = 2048
