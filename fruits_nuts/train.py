import os
import cv2 
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
register_coco_instances("fruits_nuts", {}, "../coco/data/trainval.json", "../coco/data/images")

cfg = get_cfg()
cfg.merge_from_file("/home/jyan/tmp/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# faster_rcnn_R_50_FPN_3x.yaml
cfg.DATASETS.TRAIN = ("fruits_nuts",)
cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER = (
   60
)  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
    128
)  # faster, and good enough for this toy dataset , default 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 3 classes (data, fig, hazelnut)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
