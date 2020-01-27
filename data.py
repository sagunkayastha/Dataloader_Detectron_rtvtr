import os,glob
import numpy as np
import cv2
import uuid
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
			record["file_name"] = name+'.jpg'
			record["image_id"] = name+'.jpg'+uuid.uuid4().hex[:10]
			record["height"] = height
			record["width"] = width
			objs=[]
			for obj in anno:
				
				xmin,ymin,xmax,ymax= yolo_to_voc(img,obj[0].split())
				obj= {
				'bbox': [xmin,ymin,xmax,ymax],
				'bbox_mode': BoxMode.XYXY_ABS,
				'category_id': anno[0],
				"iscrowd": 0
				}
				objs.append(dic)

		record["annotations"] = objs
		dataset_dicts.append(record)
get_dict(names)



print(dataset_dicts)
	