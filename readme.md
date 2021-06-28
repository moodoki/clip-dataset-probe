# CLIP Dataset Probe

Let's try using CLIP to look at some image datset and see what we get.,


# Setup

## Install CLIP
```
conda create -n face-dataset-analysis python=3.8
conda activate face-dataset-analysis
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

## This project's dependencies

```
pip install pandas
pip install absl-py
```


# Running


```
python query_clip.py --clip_model "ViT-B/32" --batchsize 64 --outfile out.csv --imagepath data --queries queries_strings.txt
```

Other command line options can be checked using '--help'


A CSV file will be generated with the cosine-similarity between each image and the query string.
Each column of the CSV corresponds to each line in the query strings text file.


# Known problems
Although `RN50` is an available model, using it seems to give all `nans`. Not exactly sure what's
wrong yet.
