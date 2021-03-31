## Code organization overview:

Overview of the main codes to work with rotated anchors.

### Notebooks:
- [Train](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/_train.ipynb): notebook used to train and debug the network. At the end there is code to verify the labels from anchors.
- [Inference](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/_inference.ipynb): notebook used to make visual network inferences.
- [Data generator](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/generators/convert_bb2gbb.ipynb): notebook used to create the rotated bounding boxes csv file from segmentation masks on the COCO dataset.

### Scripts:
- [Losses](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/losses.py): contain all codes for losses.
- Generator: [CSV parser](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/generators/csv_.py) and [Data parser](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/generators/common.py)
- Anchors: [Anchor matching with ProbIoU](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/utils/compute_overlap_piou.py) and [Anchor matching to labels](https://github.com/LucasKirsten/EfficientDet-unofficial/blob/rot_anchor/utils/anchors.py)
