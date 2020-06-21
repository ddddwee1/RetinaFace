# RetinaFace 

Modified from [official RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) 

## What I have done 

This is only the testing code. Not for training.

The official code takes too much CPU computation (which happens in generate_anchor:anchors_plane -> cython:anchors_cython). When running 4 processes, all cores went up to 100%. And, it can only support single image prediction. 

It is obvious that we don't need to render all anchors before selecting proposals. Therefore, I change the pipeline to:

- Select proposal indices where (conf > threshold) (image_idx, row_idx, col_idx)

- Generate anchors for selected proposals

- Generate landmarks 

- Batched_nms 

I move all operations to GPU which will be much faster. 

## Sample usage

See test_all.py

## Installation 

```
pip install torchsul 
pip install opencv-python
``` 

Other packages are already included in anaconda.

If you want to use RetinaFace-R50, download from [here](https://www.dropbox.com/s/sg84yfbobk0ql40/retina_r50.zip?dl=0)
