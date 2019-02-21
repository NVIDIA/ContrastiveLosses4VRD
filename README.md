# Graphical Contrastive Losses for Scene Graph Generation

## Setup

## Annotations

Create a data folder under the repo:
```
cd $ROOT
mkdir data
```

## OpenImages/OpenImages_mini
Download annotations for [OpenImages](https://drive.google.com/open?id=1GeUEsiS9Z3eRYnH1GPUz99wjQwjcHl6n). Unzip it under the data folder.

## Visual Genome
Download annotations for [Visual Genome](https://drive.google.com/open?id=1VDuba95vIPVhg5DiriPtwuVA6mleYGad). Unzip it under the data folder.

## Visual Relation Detection
Download annotations for [Visual Relation Detection](https://drive.google.com/open?id=1BUZIVOCEp_-_e9Rs4hVgmbKjLhR2aUT6). Unzip it under the data folder.

## Detection Models
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
