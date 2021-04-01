# detectron
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.data import MetadataCatalog, build_detection_test_loader, build_detection_train_loader

# common imports
import os
from datetime import datetime

# custom utilities
from config import ProjConfig
from data_utils import register_isaid_truck_data, get_train_aug, get_train_dataloader
from custom_trainers import AugmentedTrainer

def setup_train_config(train_data_name, val_data_name=None, output_dir=None):
    """
    Specify the model training configuration
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (train_data_name,)
    if val_data_name:
        cfg.DATASETS.TEST = (val_data_name, )
        cfg.TEST.EVAL_PERIOD = 5      # how often to eval val
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.BACKBONE.FREEZE_AT = 2   # freeze the first X stages of backbone
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.GAMMA = 0.1       # decrease LR by 1/10 every 20 steps
    cfg.SOLVER.STEPS = (20,)        # do not decay learning rate
    cfg.SOLVER.MAX_ITER = 60
    cfg.SOLVER.CHECKPOINT_PERIOD = 5 # Save a checkpoint after every this number of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # default 512, smaller numbers are faster
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    if output_dir:
        # specify an output with a few key hyper params
        cfg.OUTPUT_DIR = os.path.join(output_dir, \
            f'detectron_{datetime.now().strftime("%Y%m%d%H%M%S")}_freeze{cfg.MODEL.BACKBONE.FREEZE_AT}_batchsize{cfg.SOLVER.IMS_PER_BATCH}_lr{cfg.SOLVER.BASE_LR}')
    return cfg


def main():
    proj_config = ProjConfig()
    _ = register_isaid_truck_data(extra_meta={}, register_val=True)
    cfg = setup_train_config(
        proj_config.train_data_name,
        proj_config.val_data_name,
        proj_config.model_dir
    )
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = AugmentedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()