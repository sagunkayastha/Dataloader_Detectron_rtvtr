import os,glob
import numpy as np
import cv2
import uuid
from detectron2.structures import BoxMode\


class rtvrt_Dataloader:

	def __init__(self,img_dir,objfile):
		self.img_dir = img_dir
		self.objfile = objfile
		self.files = os.listdir(img_dir)
		self.names = []
		self.dataset_dicts=[]
		self.labels = []

		## RUN
		# self.get_names()
		# self.get_dict()
		self.get_classes()

	def get_names(self):
		for i in self.files:
			if '.txt' in i:
				self.names.append(i.split('.')[0])

	def yolo_to_voc(self,img,data,name):

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

		xmin,ymin,xmax,ymax = int(voc[0]),int(voc[1]),int(voc[2]),int(voc[3])

		return xmin,ymin,xmax,ymax

	def get_dict(self):
		names= self.names
		for pp,name in enumerate(names):
			if pp%100 == 0:
				print(pp)
			with open(self.img_dir+name+'.txt', 'r') as file:
				anno = [line.strip().split(',') for line in file]
				record = {}

				img = cv2.imread(self.img_dir+name+'.jpg')

				height, width, _ = img.shape

				record["file_name"] = self.img_dir+name+'.jpg'
				record["image_id"] = name+'.jpg'+uuid.uuid4().hex[:10]
				record["height"] = height
				record["width"] = width
				objs=[]
				for obj in anno:
					xmin,ymin,xmax,ymax= self.yolo_to_voc(img,obj[0].split(),name)
					objx= {
					'bbox': [xmin,ymin,xmax,ymax],
					'bbox_mode': BoxMode.XYXY_ABS,
					'category_id': int(obj[0].split()[0]),
					"iscrowd": 0
					}
					objs.append(objx)



			record["annotations"] = objs
			self.dataset_dicts.append(record)

	def get_classes(self):
		with open(self.objfile) as file:
			anno = [line.strip().split(',') for line in file]
			for i in anno:
				self.labels.append(i[0])
