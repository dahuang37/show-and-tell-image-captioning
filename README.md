# show-and-tell-image-captioning

## set up

```sh
# make download.sh executable
chmod +x download.sh
./download.sh 

# after data downloaded
# create vocab from train file
./datasets/build_vocab.py

# create custom valid and test data
./datasets/mscoco.py
```

