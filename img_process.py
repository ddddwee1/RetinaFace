import cv2 
import numpy as np 
import config 

def get_scale(img):
	im_shape = img.shape
	target_size = config.img_target_size
	max_size = config.img_max_size
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])
	im_scale = float(target_size) / float(im_size_min)
	if np.round(im_scale * im_size_max) > max_size:
		im_scale = float(max_size) / float(im_size_max)
	return im_scale

def process_img(img):
	oldshape = img.shape[:2]
	scale = get_scale(img)
	img = cv2.resize(img, None, fx=scale, fy=scale)
	newshape = img.shape[:2]
	borderx = (config.img_max_size - newshape[1]) // 2 
	bordery = (config.img_max_size - newshape[0]) // 2 
	canvas = np.zeros([config.img_max_size,config.img_max_size,3], dtype=np.uint8)
	canvas[bordery:bordery+newshape[0], borderx:borderx+newshape[1]] = img 
	scale = oldshape[0] / newshape[0]
	meta = [scale, borderx, bordery]
	return canvas, meta 

def postprocess_img(lmks, meta, boxes=None):
	# meta : [scale, borderx, bordery]
	for lmk in lmks:
		lmk[:,0] -= meta[1]
		lmk[:,1] -= meta[2]
		lmk[:] *= meta[0]
	if boxes is None:
		return lmks 
	else:
		for b in boxes:
			b[[0,2]] -= meta[1]
			b[[1,3]] -= meta[2]
			b[[0,1,2,3]] *= meta[0]
		return lmks, boxes

def process_batch(imgs):
	res = []
	metas = []
	for i in imgs:
		img, meta = process_img(i)
		res.append(img)
		metas.append(meta)
	res = np.array(res)
	return res, metas

def postprocess_batch(lmks, metas, boxes=None):
	if boxes is None:
		res = []
		for l,m in zip(lmks, metas):
			res.append(postprocess_img(l, m))
		return res 
	else:
		res = []
		bs = []
		for l,m,b in zip(lmks, metas, boxes):
			ll, bb = postprocess_img(l, m, b)
			res.append(ll)
			bs.append(bb)
		return res, bs 
