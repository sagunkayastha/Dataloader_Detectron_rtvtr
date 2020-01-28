from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from rtvtr_config import *
import pickle
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog



with open('data_val.p', 'rb') as fp:
	data = pickle.load(fp)
	dict_ = data
with open(objfile) as file:
	classes=[]
	anno = [line.strip().split(',') for line in file]
	for i in anno:
		classes.append(i[0])


print(len(dict_))
print("Data Loaded from Pkl")\


DatasetCatalog.register("rtvtr_val", lambda d=val: dict_)
MetadataCatalog.get("rtvtr_val").set(thing_classes=classes)
		
rtvtr_metadata = MetadataCatalog.get("rtvtr_val")

predictor = DefaultPredictor(cfg)




cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("rtvtr_val", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode

for d in random.sample(dict_, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=rtvtr_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(v.get_image()[:, :, ::-1])