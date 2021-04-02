# detectron
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import DefaultPredictor

# common imports
import os
import numpy as np
import argparse
import json

# custom utilities
from config import ProjConfig
from data_utils import register_isaid_truck_data


def setup_eval_config(model_path, test_data_name):
    """
    Specify the model evaluation configuration

    Args:
        model_path (str): path to the trained model, e.g. '/path/to/file.pth'

    Returns:
        configuration
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TEST = (test_data_name, )
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_path
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # default 512, smaller numbers are faster
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    return cfg


def predict_dataloader(model, dataloader):
    """
    A custom predict function using native pytorch

    Returns:
        a list of dict where each dict is the predictions for an images. Sample structure shown
        below:
            {'instances': Instances(
                num_instances=0,
                image_height=1023,
                image_width=1147,
                fields=[
                    pred_boxes: Boxes(tensor([], size=(0, 4))),
                    scores: tensor([]),
                    pred_classes: tensor([], dtype=torch.int64),
                    pred_masks: tensor([], size=(0, 1023, 1147), dtype=torch.uint8)
                    ])}
    """
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            all_preds.append(model(batch)[0])
    return all_preds


def predict_single_image(image, cfg):
    """
    Args:
        image: a BGR image
        cfg: config

    Returns:
        a dictionary of output for the input image. Refer to the sample structure listed in
        predict_dataloader.
    """
    predictor = DefaultPredictor(cfg)
    outputs = pred(inputs)


def main(args):
    # configure the data
    proj_config = ProjConfig()
    _ = register_isaid_truck_data(extra_meta={}, register_val=False, register_test=True)

    cfg = setup_eval_config(args.model_path, proj_config.test_data_name)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(proj_config.test_data_name, distributed=args.distributed, output_dir=args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    val_dataloader = build_detection_test_loader(cfg, proj_config.test_data_name)

    # load the trained model
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    # run inference and evaluation, saving predictions and eval results to output_dir
    eval_result = inference_on_dataset(model, val_dataloader, evaluator)

    # # alternatively, if only need inference result
    # all_preds = predict(model, val_dataloader)
    # with open(os.path.join(args.output_dir, 'instances_predictions.json'), 'w') as fp:
    #     json.dump(all_preds, fp)


# models/detectron_20210401155414_freeze2_batchsize2_lr0.00025/model_0000009.pth
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--output_dir', type=str)
    args, _ = parser.parse_known_args()

    main(args)