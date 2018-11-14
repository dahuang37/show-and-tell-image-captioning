echo "Downloading COCO dataset"
# create folder and downloads coco data
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P ./data/coco
wget http://images.cocodataset.org/zips/train2014.zip -P ./data/coco
wget http://images.cocodataset.org/zips/val2014.zip -P ./data/coco
# wget http://images.cocodataset.org/zips/test2014.zip -P ./data/coco
# wget http://images.cocodataset.org/annotations/image_info_test2014.zip -P ./data/coco

echo "unziping data"
unzip ./data/coco/captions_train-val2014.zip -d ./data/coco
rm ./data/coco/captions_train-val2014.zip
unzip ./data/coco/train2014.zip -d ./data/coco
rm ./data/coco/train2014.zip 
unzip ./data/coco/val2014.zip -d ./data/coco
rm ./data/coco/val2014.zip 
# unzip ./data/coco/test2014.zip -d ./data/coco
# rm ./data/coco/test2014.zip 
# unzip ./data/coco/image_info_test2014.zip -d ./data/coco
# rm ./data/coco/image_info_test2014.zip 
echo "Download complete"