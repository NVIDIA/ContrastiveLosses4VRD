# Graphical Contrastive Losses for Scene Graph Generation

% ## Setup

## Annotations

Create a data folder under the repo:
```
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
Create an folder for the training images:
```
cd $ROOT/data/openimages_v4
mkdir train
```
Download OpenImages v4 training images from the [official page](https://storage.googleapis.com/openimages/web/download.html). **Note:** only training images are needed since our annotations will split them into a train and a validation set. Put all images in `train/`

### Visual Genome
Create an folder for all images:
```
cd $ROOT/data/vg
mkdir VG_100K
```
Download Visual Genome images from the [official page](https://visualgenome.org/api/v0/api_home.html). Unzip all images into `VG_100K/`.

### Visual Relation Detection
Create an folder for train and validation images:
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

## References
