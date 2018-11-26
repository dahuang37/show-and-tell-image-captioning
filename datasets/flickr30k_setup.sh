echo "Remember to set flickr dataset under data/flickr8k/Flickr8k_text  and data/flickr8k/Flicker8k_Dataset"
echo "Building datasets json file for flickr8k"
python flickr8k.py 
# create custom valid and test data
echo "Building vocab for flickr8k"
python build_vocab.py --json ../data/flickr30k/buildvocab.json --file_path ../data/flickr30k/vocab.pkl