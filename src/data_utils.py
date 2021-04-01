from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
from detectron2.data import build_detection_test_loader, build_detection_train_loader

# custom imports
from config import ProjConfig


def register_isaid_truck_data(extra_meta={}, register_val=True, register_test=False):
    """
    register project data with name isaid_truck_train/val
    """
    proj_config = ProjConfig()
    extra_meta = extra_meta     # additional metadata to be associated w/ dataset
    register_coco_instances(proj_config.train_data_name,
                            extra_meta,
                            proj_config.train_anno_path,
                            proj_config.train_image_path)
    if register_val:
        register_coco_instances(proj_config.val_data_name,
                                extra_meta,
                                proj_config.val_anno_path,
                                proj_config.val_image_path)
    if register_test:
        register_coco_instances(proj_config.test_data_name,
                                extra_meta,
                                proj_config.test_anno_path,
                                proj_config.test_image_path)

    return True


def get_train_aug():
    """
    A chain of random data augmentation to be applied at training.
    Parameter values picked to yield realistic images after transformation.

    Returns:
        A list of detectron2.data.transforms.Augmentation
    """
    custom_aug = [
        T.RandomBrightness(.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomApply(T.RandomRotation(angle=90, expand=True), prob=0.25),
        T.RandomApply(T.RandomRotation(angle=270, expand=True), prob=0.25),
        # To avoid losing information after rotation, specify to expand canvas.
        T.RandomApply(T.RandomRotation(angle=[-30, 30], expand=True), prob=0.125)
    ]
    return custom_aug


def get_train_dataloader(cfg, aug):
    """
    Args:
        cfg: the configuration
        aug (list): a list of data augmentations

    Returns:
        A custom dataloader with augmentation specified
    """
    train_mapper = DatasetMapper(cfg, is_train=True, augmentations=aug)
    dataloader = build_detection_train_loader(cfg, mapper=train_mapper)
    return dataloader