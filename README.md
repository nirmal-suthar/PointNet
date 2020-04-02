# PointNet
Implementation of PointNet classification in Flux machine learning library written in Julia.

For downloading and processing the modelnet40 dataset.
```bash
$ URL=https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
$ ZIP_FILE=./data/modelnet40.zip
$ mkdir -p ./data/
$ wget -N $URL -O $ZIP_FILE
$ unzip $ZIP_FILE -d ./data/
$ rm $ZIP_FILE
```
    
For running the model

```bash
$ julia --project="." PointNet.jl
```

## Results

Both model are trained with similar config
```
batch_size = 32
npoints = 1024
lr = 0.003 (ADAM)
K ( nearest-neighbour, used in DGCNN ) = 10
```

Dataset | ModelNet (10 classes) |
---|--|
DGCNN |<b>0.8744</b>|
PointNet |0.7544|
