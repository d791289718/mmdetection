from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class DOTADataset(CocoDataset):

    CLASSES = ("plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout", "soccer-ball-field", "swimming-pool", "container-crane")