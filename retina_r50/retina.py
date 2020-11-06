import numpy as np 
import torch
from TorchSUL import Model as M 
import retina_resnet 
import cv2 
import config 
import torchvision 

def _whctrs(anchor):
	w = anchor[2] - anchor[0] + 1
	h = anchor[3] - anchor[1] + 1
	x_ctr = anchor[0] + 0.5 * (w - 1)
	y_ctr = anchor[1] + 0.5 * (h - 1)
	return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
	ws = ws[:, np.newaxis]
	hs = hs[:, np.newaxis]
	anchors = np.hstack((x_ctr-0.5*(ws-1), y_ctr-0.5*(hs-1), x_ctr+0.5*(ws-1), y_ctr+0.5*(hs-1)))
	return anchors

def _ratio_enum(anchor, ratios):
	w, h, x_ctr, y_ctr = _whctrs(anchor)
	size = w * h
	size_ratios = size / ratios
	ws = np.round(np.sqrt(size_ratios))
	hs = np.round(ws * ratios)
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors

def _scale_enum(anchor, scales):
	w, h, x_ctr, y_ctr = _whctrs(anchor)
	ws = w * scales
	hs = h * scales
	anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
	return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=2**np.arange(3, 6), stride=16, dense_anchor=False):
	base_anchor = np.array([1, 1, base_size, base_size]) - 1
	ratio_anchors = _ratio_enum(base_anchor, ratios)
	anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales) for i in range(ratio_anchors.shape[0])])
	if dense_anchor:
		assert stride%2==0
		anchors2 = anchors.copy()
		anchors2[:,:] += int(stride/2)
		anchors = np.vstack( (anchors, anchors2) )
	return anchors

def generate_anchors_fpn(dense_anchor=False):
	cfg = {
		'32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': (1.,), 'ALLOWED_BORDER': 9999},
		'16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': (1.,), 'ALLOWED_BORDER': 9999},
		'8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': (1.,), 'ALLOWED_BORDER': 9999}
	}

	RPN_FEAT_STRIDE = []
	for k in cfg:
	  RPN_FEAT_STRIDE.append(int(k))
	RPN_FEAT_STRIDE = sorted(RPN_FEAT_STRIDE, reverse=True)

	anchors = []
	for k in RPN_FEAT_STRIDE:
		v = cfg[str(k)]
		bs = v['BASE_SIZE']
		__ratios = np.array(v['RATIOS'])
		__scales = np.array(v['SCALES'])
		stride = int(k)
		r = generate_anchors(bs, __ratios, __scales, stride, dense_anchor)
		anchors.append(r)
	return anchors

def _clip_pad(tensor, pad_shape):
	H, W = tensor.shape[2:]
	h, w = pad_shape
	if h < H or w < W:
		tensor = tensor[:, :, :h, :w].copy()
	return tensor

def bbox_pred(boxes, box_deltas):
	if boxes.shape[0] == 0:
		# return np.zeros((0, box_deltas.shape[1]))
		return torch.zeros(0, box_deltas.shape[1], device='cuda')

	# boxes = boxes.astype(np.float, copy=False)
	widths = boxes[:, 2:3] - boxes[:, 0:1] + 1.0
	heights = boxes[:, 3:4] - boxes[:, 1:2] + 1.0
	ctr_x = boxes[:, 0:1] + 0.5 * (widths - 1.0)
	ctr_y = boxes[:, 1:2] + 0.5 * (heights - 1.0)

	dx = box_deltas[:, 0:1]
	dy = box_deltas[:, 1:2]
	dw = box_deltas[:, 2:3]
	dh = box_deltas[:, 3:4]

	pred_ctr_x = dx * widths + ctr_x
	pred_ctr_y = dy * heights + ctr_y
	pred_w = torch.exp(dw) * widths
	pred_h = torch.exp(dh) * heights

	
	pred_boxes = []
	pred_boxes.append(pred_ctr_x - 0.5 * (pred_w - 1.0))
	pred_boxes.append(pred_ctr_y - 0.5 * (pred_h - 1.0))
	pred_boxes.append(pred_ctr_x + 0.5 * (pred_w - 1.0))
	pred_boxes.append(pred_ctr_y + 0.5 * (pred_h - 1.0))
	pred_boxes = torch.cat(pred_boxes, dim=1)

	# pred_boxes = torch.zeros(box_deltas.shape)
	# pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
	# pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
	# pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
	# pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

	if box_deltas.shape[1]>4:
		pred_boxes[:, 4:] = box_deltas[:, 4:]
	return pred_boxes

def clip_boxes(boxes, im_shape):
	# boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1]-1), 0)
	# boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0]-1), 0)
	# boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1]-1), 0)
	# boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0]-1), 0)
	boxes[:, [0,2]] = torch.clamp(boxes[:, [0,2]], 0, im_shape[1]-1)
	boxes[:, [1,3]] = torch.clamp(boxes[:, [1,3]], 0, im_shape[0]-1)
	return boxes

def landmark_pred(boxes, landmark_deltas):
	if boxes.shape[0] == 0:
		# return np.zeros((0, landmark_deltas.shape[1], 2))
		return torch.zeros(0, landmark_deltas.shape[1], 2, device='cuda')
	# boxes = boxes.astype(np.float, copy=False)
	widths = boxes[:, 2] - boxes[:, 0] + 1.0
	heights = boxes[:, 3] - boxes[:, 1] + 1.0
	ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
	ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)


	# pred = landmark_deltas.copy()
	# for i in range(5):
	# 	pred[:, i, 0] = landmark_deltas[:, i, 0] * widths + ctr_x
	# 	pred[:, i, 1] = landmark_deltas[:, i, 1] * heights + ctr_y

	whs = torch.stack([widths, heights], dim=1).unsqueeze(1)
	ctrs = torch.stack([ctr_x, ctr_y], dim=1).unsqueeze(1)
	pred = landmark_deltas * whs + ctrs
	return pred

def post_proc(net_out, im_info, threshold, im_scale, _feat_stride_fpn, _num_anchors, _anchors_fpn):
	proposals_list = []
	scores_list = []
	landmarks_list = []
	imgidx_list = []
	for _idx, s in enumerate(_feat_stride_fpn):

		stride = int(s)
		idx = _idx * 3
		scores = net_out[idx]

		scores = scores[:, _num_anchors['stride%s'%s]:, :, :]

		idx += 1
		bbox_deltas = net_out[idx]

		height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

		A = _num_anchors['stride%s'%s]
		K = height * width
		anchors_fpn = _anchors_fpn['stride%s'%s]
		
		scores = torch.einsum('ijkl->iklj', scores).reshape(-1)
		
		order = torch.where(scores>=threshold)[0]

		bbox_deltas = torch.einsum('ijkl->iklj', bbox_deltas)
		bbox_pred_len = bbox_deltas.shape[3]//A
		bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
		
		pixidx = torch.div(order, 2)
		category = torch.remainder(order, 2)
		pixrow = torch.div(pixidx, width)
		pixcol = torch.remainder(pixidx, width)
		imgidx = torch.div(pixrow, height)
		pixrow = torch.remainder(pixrow, height)
		
		pixrow = pixrow.unsqueeze(-1)
		pixcol = pixcol.unsqueeze(-1)

		anchors = anchors_fpn[category]
		anchors[:, [0,2]] += s * pixcol
		anchors[:, [1,3]] += s * pixrow
		proposals = bbox_pred(anchors, bbox_deltas[order])
		proposals = clip_boxes(proposals, im_info[:2])

		scores = scores[order]

		proposals[:,0:4] /= im_scale
		proposals_list.append(proposals)
		scores_list.append(scores)
		imgidx_list.append(imgidx)

		idx+=1
		landmark_deltas = net_out[idx]
		landmark_pred_len = landmark_deltas.shape[1]//A
		landmark_deltas = torch.einsum('ijkl->iklj', landmark_deltas).reshape((-1, 5, landmark_pred_len//5))
		landmarks = landmark_pred(anchors, landmark_deltas[order])
		landmarks[:, :, 0:2] /= im_scale
		landmarks_list.append(landmarks)

	proposals = torch.cat(proposals_list, dim=0)
	landmarks = torch.cat(landmarks_list, dim=0)
	scores = torch.cat(scores_list)
	imgidx = torch.cat(imgidx_list)

	return proposals, landmarks, scores, imgidx


class RetinaFace(object):
	def __init__(self, modelpath, nms=0.4, worker=2):
		model = retina_resnet.Detector()
		model = model.eval()
		x = torch.from_numpy(np.ones([1,3,640,640]).astype(np.float32))
		_ = model(x)
		M.Saver(model).restore(modelpath)
		model.cuda()
		if isinstance(config.gpus, list):
			if len(config.gpus)>1:
				print('Using multiple gpus:', config.gpus)
				model = torch.nn.DataParallel(model, device_ids=config.gpus)
		self.model = model

		self.nms_threshold = nms

		self.fpn_keys = []
		self._feat_stride_fpn = [32, 16, 8]

		for s in self._feat_stride_fpn:
			self.fpn_keys.append('stride%s'%s)

		self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=False)))
		for k in self._anchors_fpn:
			v = self._anchors_fpn[k].astype(np.float32)
			v = torch.from_numpy(v).cuda()
			self._anchors_fpn[k] = v

		self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
		self.worker = worker
		
	def detect(self, img_list, threshold=0.6, keep_first=False):
		# im_scale=scale
		batch_size = len(img_list)
		im_info = [img_list[0].shape[0], img_list[0].shape[1]]
		im_scale = 1.0

		inputs = np.stack(img_list, axis=0).astype(np.float32).transpose((0, 3, 1, 2))
		inputs = inputs[:,::-1].copy()
		inputs = torch.from_numpy(inputs).cuda()

		with torch.no_grad():
			net_outs = self.model(inputs)

		net_out = []
		for layer_idx in range(len(net_outs)):
			net_out.append(net_outs[layer_idx])

		result = post_proc(net_out, im_info, threshold, im_scale, self._feat_stride_fpn, self._num_anchors, self._anchors_fpn)

		proposals, landmarks, scores, imgidx = result
		if proposals.shape[0]>0:
			# order = torch.argsort(scores, descending=True)
			# proposals = proposals[order]
			# landmarks = landmarks[order]
			# scores = scores[order]
			# imgidx = imgidx[order]
			keep = torchvision.ops.boxes.batched_nms(proposals, scores, imgidx, self.nms_threshold)
			#keep = keep.cpu().numpy()

			proposals = torch.cat([proposals, scores.unsqueeze(1)], dim=1)
			landmarks = landmarks[keep].cpu().detach().numpy()
			scores = scores[keep].cpu().detach().numpy()
			proposals = proposals[keep, :].cpu().detach().numpy()
			imgidx = imgidx[keep].cpu().detach().numpy()

			# order = torch.argsort(scores, descending=True)
			# proposals = proposals[order].cpu().numpy()
			# landmarks = landmarks[order].cpu().numpy()
			# imgidx = imgidx[order].cpu().numpy()

			faces_batch = [None for i in range(batch_size)]
			landmarks_batch = [None for i in range(batch_size)]
			for i in range(imgidx.shape[0]):
				idx = imgidx[i]
				if faces_batch[idx] is None:
					faces_batch[idx] = []
					landmarks_batch[idx] = []
				faces_batch[idx].append(proposals[i])
				landmarks_batch[idx].append(landmarks[i])

			if keep_first:
				# faces_batch = [[i[0]] for i in faces_batch if i is not None else None]
				# landmarks_batch = [[i[0]] for i in landmarks_batch if i is not None else None]
				faces_batch_2 = []
				landmarks_batch_2 = []
				for i,l in zip(faces_batch, landmarks_batch):
					if i is None:
						faces_batch_2.append(None)
						landmarks_batch_2.append(None)
					else:
						faces_batch_2.append([i[0],])
						landmarks_batch_2.append([l[0],])
				faces_batch = faces_batch_2
				landmarks_batch = landmarks_batch_2
		else:
			faces_batch = landmarks_batch = [None] * batch_size

		return faces_batch, landmarks_batch
