# Setup


## Install CLIP
```
conda create -n face-dataset-analysis python=3.8
conda activate face-dataset-analysis
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

```
pip install pandas
pip install absl-py
```
