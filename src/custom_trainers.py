from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader


class TrainerWithVal(DefaultTrainer):
    """
    This is a customized Trainer that runs evaluation on cfg.DATASETS.TEST (validation set)
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """class method for evaluating the validation set"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


class AugmentedTrainer(TrainerWithVal):
    """
    This is a customized Trainer with custom augmentation applied.
    1. Creates model, optimizer, scheduler, dtaloader from the given config, where the
       dataloader has the custom data augmentation
    2. Run evaluation on cfg.DATASETS.TEST
    """
    @classmethod
    def build_train_loader(cls, cfg):
        train_aug = get_train_aug()
        train_dataloader = get_train_dataloader(cfg, train_aug)
        return train_dataloader

