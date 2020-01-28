from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os,cv2
from rtvtr_config import *
import pickle,random
from detectron2.engine import DefaultPredictor

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2 import model_zoo


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


DatasetCatalog.register("rtvtr_val", lambda d='val': dict_)
MetadataCatalog.get("rtvtr_val").set(thing_classes=classes)

rtvtr_metadata = MetadataCatalog.get("rtvtr_val")






cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0004999.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("rtvtr_val", )
predictor = DefaultPredictor(cfg)

print(type(predictor))
from detectron2.utils.visualizer import ColorMode
i=0
for d in dict_:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=rtvtr_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    i+=1
	cv2.imwrite('AI/test_'+str(i)+'.jpg',vis.get_image()[:, :, ::-1])