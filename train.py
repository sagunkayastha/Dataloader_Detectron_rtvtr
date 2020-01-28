import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os,glob
import numpy as np
import cv2
import uuid
import random
from detectron2.structures import BoxMode
img_dir = 'to_send/'
files = os.listdir(img_dir)


names = []
dataset_dicts = []

for i in files:
	if '.txt' in i:
		tx = i.split('.')[0]
		names.append(tx)

def yolo_to_voc(img,data):
	
	
	
	height, width, _ = img.shape

	
	
	voc = []
	bbox_width = float(data[3]) * width
	bbox_height = float(data[4]) * height
	center_x = float(data[1]) * width
	center_y = float(data[2]) * height
	voc.append(center_x - (bbox_width / 2))
	voc.append(center_y - (bbox_height / 2))
	voc.append(center_x + (bbox_width / 2))
	voc.append(center_y + (bbox_height / 2))
	# voc_labels.append(voc)

	xmin,ymin,xmax,ymax = int(voc[0]),int(voc[1]),int(voc[2]),int(voc[3])
	
	
	return xmin,ymin,xmax,ymax

def get_dict(names):
	for name in names:
		
		with open(img_dir+name+'.txt', 'r') as file:
			anno = [line.strip().split(',') for line in file]
			record = {}
			
			img = cv2.imread('to_send/'+name+'.jpg')
			height, width, _ = img.shape
			record["file_name"] = img_dir+name+'.jpg'
			record["image_id"] = name+'.jpg'+uuid.uuid4().hex[:10]
			record["height"] = height
			record["width"] = width
			objs=[]
			for obj in anno:
				
				xmin,ymin,xmax,ymax= yolo_to_voc(img,obj[0].split())
				
				obj= {
				'bbox': [xmin,ymin,xmax,ymax],
				'bbox_mode': BoxMode.XYXY_ABS,
				'category_id': int(obj[0][0]),
				"iscrowd": 0
				}
				objs.append(obj)

		record["annotations"] = objs
		dataset_dicts.append(record)
		return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog

classes_o = ['car','bus','motorbike','pickup','license_plate','taxi','state_plate','truck','van','scooter','micro_bus','tripper',
'heavy_equipment_vehicle',
'tempo',
'police_vehicle',
'embossed_plate',
'firetruck',
'ambulance']

classes=[str(i) for i in range(18)]
# print(classes)
DatasetCatalog._REGISTERED.clear()

for d in ["train", "val"]:
    DatasetCatalog.register("rtvtr_" + d, lambda d=d: get_dict(names))
    MetadataCatalog.get("rtvtr_" + d).set(thing_classes=classes)
rtvtr_metadata = MetadataCatalog.get("rtvtr_train")

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("rtvtr_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 1
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 18  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()