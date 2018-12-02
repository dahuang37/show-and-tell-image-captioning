echo "Remember to set flickr dataset under data/flickr30k/"
echo "Building datasets json file for flickr30k"
python3 datasets/flickr30k.py
# create custom valid and test data
echo "Building vocab for flickr30k"
python3 datasets/build_vocab.py --json ./data/flickr30k/buildvocab.json --file_path ./data/flickr30k/vocab.pkl