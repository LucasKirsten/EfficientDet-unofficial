# train head layers
python train.py \
--snapshot imagenet \
--snapshot-path checkpoints/gbb_l3 \
--loss piou_l3 \
--regression_weight 1 \
--phi 0 \
--weighted-bifpn \
--gpu 0,1 \
--epochs 50 \
--freeze-backbone \
--no-evaluation \
--compute-val-loss \
--lr 1e-3 \
--batch-size 16 \
--steps_per_epoch 1000 \
csv --annotations_path /datasets/dataset/coco2017/annotations/instances_train2017_obb_v2.csv \
--base_dir_train /datasets/dataset/coco2017/train2017 \
--val_annotations_path /datasets/dataset/coco2017/annotations/instances_val2017_obb_v2.csv \
--base_dir_val /datasets/dataset/coco2017/val2017 \
--classes_path /datasets/dataset/coco2017/annotations/classes_voc.csv

# train all layers
python train.py \
--snapshot checkpoints/gbb_l3/csv.h5 \
--snapshot-path checkpoints/gbb_l3/gbb_finetuned \
--loss piou_l1 \
--regression_weight 2 \
--phi 0 \
--weighted-bifpn \
--gpu 0,1 \
--epochs 100 \
--no-evaluation \
--compute-val-loss \
--lr 1e-3 \
--batch-size 16 \
--steps_per_epoch 1000 \
csv --annotations_path /datasets/dataset/coco2017/annotations/instances_train2017_obb_v2.csv \
--base_dir_train /datasets/dataset/coco2017/train2017 \
--val_annotations_path /datasets/dataset/coco2017/annotations/instances_val2017_obb_v2.csv \
--base_dir_val /datasets/dataset/coco2017/val2017 \
--classes_path /datasets/dataset/coco2017/annotations/classes_voc.csv