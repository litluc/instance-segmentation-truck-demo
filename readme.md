# Commercial Truck Detection

A repository for work done on data preparation and model-building for commercial truck detection.

Below are high level description of the project:

- Target: detection and masking all commercial trucks from aerial images
- Data and labels: [iSAID](https://captain-whu.github.io/iSAID/) data with COCO labels, 'Large Vehicle' category only
- Model architecture: [Mask RCNN](https://arxiv.org/abs/1703.06870)
- Implementation: Detectron2


---

## Detectron2
[Detectron2](https://github.com/facebookresearch/detectron2) is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.
It is a ground-up rewrite of the previous version,
[Detectron](https://github.com/facebookresearch/Detectron/),
and it originates from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/).

<div align="center">
  <img src="https://user-images.githubusercontent.com/1381301/66535560-d3422200-eace-11e9-9123-5535d469db19.png"/>
</div>