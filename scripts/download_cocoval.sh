#!/bin/bash
mkdir -p data
cd data
pwd

echo "Getting annotations"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
wget -c $ANN_URL
unzip -f annotations_trainval2017.zip
rm annotations/captions_* annotations/person_* annotations/instances_train2017.json 

echo "Getting images"
IMG_URL="http://images.cocodataset.org/zips/val2017.zip"
wget -c $IMG_URL
unzip -f val2017.zip