# create dir 
mkdir data
mkdir ./data/coco ./data/sbu ./data/pascal ./data/flickr30k ./data/flickr8k ./checkpoints

# download data
chmod +x download.sh
[ "$(ls -A ./data/coco)" ] && echo "COCO data downloaded" || ./download.sh

# after data downloaded
# create vocab from train file
echo "Building vocab for mscoco"
python ./datasets/build_vocab.py

# create custom valid and test data
echo "Spliting mscoco val data to test"
python ./datasets/mscoco.py

 

