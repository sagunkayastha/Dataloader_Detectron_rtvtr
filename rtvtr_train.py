# import some common libraries
import numpy as np
import cv2
import random
import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import pickle



# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


### c####



from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo


from dataloader import rtvrt_Dataloader
from rtvtr_config import *
def main():
	Data = rtvrt_Dataloader(img_dir, objfile)

	if Load_from_file == False:
		Data.get_names()
		Data.get_dict()
		dict_ = Data.dataset_dicts
		classes = Data.labels
		print("Data Loaded from folder")
		with open('data.p', 'wb') as fp:
			pickle.dump(dict_, fp, protocol=pickle.HIGHEST_PROTOCOL)

	else:
		with open('data.p', 'rb') as fp:
			data = pickle.load(fp)
		dict_ = data
		classes = Data.labels
		print(len(dict_))
		print("Data Loaded from Pkl")

	for d in ["train", "val"]:
		DatasetCatalog.register("rtvtr_" + d, lambda d=d: dict_)
		MetadataCatalog.get("rtvtr_" + d).set(thing_classes=classes)

	rtvtr_metadata = MetadataCatalog.get("rtvtr_train")





	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
	cfg.DATASETS.TRAIN = ("rtvtr_train",)
	cfg.DATASETS.TEST = ("rtvtr_val")
	cfg.DATALOADER.NUM_WORKERS = 4
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
	cfg.SOLVER.MAX_ITER = 50000
	cfg.SOLVER.STEPS = (5000)
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # (default: 512)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18
	# cfg.OUTPUT_DIR = "./checkpoints/"
	cfg.SOLVER.CHECKPOINT_PERIOD = 500
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = DefaultTrainer(cfg)
	trainer.resume_or_load(resume=resume)
	trainer.train()

if __name__ == "__main__":
	main()