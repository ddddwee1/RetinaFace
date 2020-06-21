import sys 
import config 
sys.path.append(config.detector_path)
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import retina
import img_preprocess
import glob 

def visualize_img(img, lmk, box=None):
	if lmk is None:
		return img 
	img = img.copy()
	for i in range(len(lmk)):
		l = lmk[i]
		for j in range(5):
			cv2.circle(img, (int(l[j,0]), int(l[j,1])), 3, (0,0,255), -1)
		if box is not None:
			b = box[i]
			cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0,0,255), 3)
	return img 

def visualize(imgs, lmks, boxes=None):
	res = []
	for i in range(len(imgs)):
		im = visualize_img(imgs[i], lmks[i], None if boxes is None else boxes[i])
		res.append(im)
	return res 

if __name__=="__main__":
	pth = config.detector_path+'model_detection/'
	detector = retina.RetinaFace(pth, nms=config.nms_thresh)

	if not os.path.exists('./results/'):
		os.mkdir('./results/')

	img_paths = glob.glob('./imgs/*.*')
	imgs = [cv2.imread(i) for i in img_paths]
	img_batch, metas = img_preprocess.process_batch(imgs)

	boxes, lmks = detector.detect(img_batch, threshold=config.box_score_thresh)
	lmks, boxes = img_preprocess.postprocess_batch(lmks, metas, boxes)

	vis = visualize(imgs, lmks, boxes)
	for i in range(len(vis)):
		cv2.imwrite('./results/%d.jpg'%i, vis[i])
