# Point-Cloud

### Download pretrained Weights
**ResNeXt101**: https://pan.baidu.com/s/1o2oVMiLRu770Fdpa65Pdbw?pwd=g3yi

**SuperPoint**: https://github.com/SabihShah/Point-Cloud/blob/main/superpoint_v1.pth


### Inference:
run
```
python "main.py" --rgb_path 'path to rgb image' --load_ckpt res101.pth 
```
for generating a point cloud from a single rgb image

run
``` 
python main.py --rgb_path 'path to rgb image' --load_ckpt res101.pth --smooth --use_keypoints --keypoint_weight superpoint_v1.pth --show_difference
```
for generating a point cloud and smoothing using KNN and keypoints

*Remove the --use_keypoints and --keypoint_weight arguments to use **RANSAC** for outlier detection and removal*

Add output directory argument to save the results


Other arguments can be set such as camera's intrinsic parameters if known for better results
