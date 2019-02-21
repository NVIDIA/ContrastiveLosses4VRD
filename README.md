# Graphical Contrastive Losses for Scene Graph Generation

## Annotations

Create a data folder at the top-level directory of the repository:
```
export ROOT=path/to/cloned/repository
cd $ROOT
mkdir data
```

### OpenImages/OpenImages_mini
Download it [here](https://drive.google.com/open?id=1GeUEsiS9Z3eRYnH1GPUz99wjQwjcHl6n). Unzip it under the data folder. You should see an `openimages_v4` folder unzipped there.

### Visual Genome
Download it [here](https://drive.google.com/open?id=1VDuba95vIPVhg5DiriPtwuVA6mleYGad). Unzip it under the data folder. You should see a `vg` folder unzipped there.

### Visual Relation Detection
Download it [here](https://drive.google.com/open?id=1BUZIVOCEp_-_e9Rs4hVgmbKjLhR2aUT6). Unzip it under the data folder. You should see a `vrd` folder unzipped there.

## Images

### OpenImages
Create a folder for the training images:
```
cd $ROOT/data/openimages_v4
mkdir train
```
Download OpenImages v4 training images from the [official page](https://storage.googleapis.com/openimages/web/download.html). **Note:** only training images are needed since our annotations will split them into a train and a validation set. Put all images in `train/`

### Visual Genome
Create a folder for all images:
```
cd $ROOT/data/vg
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images (part 1 and part 2) into `VG_100K/`. There should be a total of 108249 files.

### Visual Relation Detection
Create a folder for train and validation images:
```
cd $ROOT/data/vrd
mkdir train_images
mkdir val_images
```
Download Visual Relation Detection images from the [official page](https://cs.stanford.edu/people/ranjaykrishna/vrd/). Put training images into `train_images/` and testing images into `val_images/`.

## Pre-trained Detection Models
Download pre-trained detection models [here](https://drive.google.com/open?id=1_7Qw8oqDvmMpp9cBCkUZY7PByH6iINOl). Unzip it under the root directory.

## Directory Structure
The final directories for data and detection models should look like:
```
|-- detection_models
|   |-- oi_rel
|   |   |-- X-101-64x4d-FPN
|   |   |   |-- model_step599999.pth
|   |-- vg
|   |   |-- VGG16
|   |   |   |-- model_step479999.pth
|   |   |-- X-101-64x4d-FPN
|   |   |   |-- model_step119999.pth
|   |-- vrd
|   |   |-- VGG16
|   |   |   |-- model_step4499.pth
|-- data
|   |-- openimages_v4
|   |   |-- rel
|   |   |-- train    <-- (contains OpenImages_v4 training/validation images)
|   |   |   |-- rel_only_annotations_train.json
|   |   |   |-- rel_only_annotations_val.json
|   |   |   |-- ...
|   |-- vg
|   |   |-- VG_100K    <-- (contains Visual Genome all images)
|   |   |-- rel_annotations_train.json
|   |   |-- rel_annotations_val.json
|   |   |-- ...
|   |-- vrd
|   |   |-- train_images    <-- (contains Visual Relation Detection training images)
|   |   |-- val_images    <-- (contains Visual Relation Detection validation images)
|   |   |-- new_annotations_train.json
|   |   |-- new_annotations_val.json
|   |   |-- ...
```

## Evaluating Pre-trained models

## Training from scratch

### OpenImages_mini
To train our network using a ResNeXt-101-64x4d-FPN backbone, run
```
python tools/train_net_step_rel.py --dataset oi_rel_mini --cfg configs/oi_rel_mini/e2e_faster_rcnn_X-101-64x4d-FPN_12_epochs_oi_rel_mini_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5.yaml --nw 8 --use_tfboard
```

### OpenImages
To train our network using a ResNeXt-101-64x4d-FPN backbone, run
```
python tools/train_net_step_rel.py --dataset oi_rel --cfg configs/oi_rel/e2e_faster_rcnn_X-101-64x4d-FPN_12_epochs_oi_rel_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5.yaml --nw 8 --use_tfboard
```

### Visual Genome
To train our network using a VGG16 backbone, run
```
python tools/train_net_step_rel.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_VGG16_8_epochs_vg_v3_default_two_level_use_so_separate_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_no_spt.yaml --nw 8 --use_tfboard
```
To train our network using a ResNeXt-101-64x4d-FPN backbone, run
```
python tools/train_net_step_rel.py --dataset vg --cfg configs/vg/e2e_faster_rcnn_X-101-64x4d-FPN_8_epochs_vg_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5.yaml --nw 8 --use_tfboard
```

### Visual Relation Detection
To train our network initialized by an ImageNet pre-trained VGG16 model, run
```
python tools/train_net_step_rel.py --dataset vrd --cfg configs/vrd/e2e_faster_rcnn_VGG16_16_epochs_vrd_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_IN_pretrained.yaml --nw 8 --use_tfboard
```
To train our network initialized by a COCO pre-trained VGG16 model, run
```
python tools/train_net_step_rel.py --dataset vrd --cfg configs/vrd/e2e_faster_rcnn_VGG16_16_epochs_vrd_v3_default_node_contrastive_loss_w_so_p_aware_margin_point2_so_weight_point5_COCO_pretrained.yaml --nw 8 --use_tfboard
```

## References
