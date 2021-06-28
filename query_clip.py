import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import clip

from pathlib import Path

from absl import app, flags
from absl import logging

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

__available_models = clip.available_models()
__default_device = "cuda" if torch.cuda.is_available() else "cpu"

FLAGS = flags.FLAGS
flags.DEFINE_string('queries', 'query_strings.txt', 'Text file containing query strings')
flags.DEFINE_string('prepend', 'a photo of a', 'text to prepend to the strings')
flags.DEFINE_string('imagepath', 'data', 'path to data')
flags.DEFINE_string('outfile', 'out.csv', 'output filename')
flags.DEFINE_enum('clip_model', __available_models[0], __available_models, 'which clip model to use')
flags.DEFINE_string('device', __default_device, 'which torch device to use')
flags.DEFINE_integer('batchsize', 16, 'batchsize')


class ImageFolderFilename(ImageFolder):
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, path) where path is the path to the file
        """
                 # Get the file path and the serial number of the folder to which it belongs
        path, target = self.samples[index]
                 # Download Data 
        sample = self.loader(path)
                 # Whether to process the read data
                 # Mainly includes methods for converting tensors and some data enhancements
        if self.transform is not None:
            sample = self.transform(sample)
                 # Whether to process the serial number of the folder to which it belongs
                 # Since torchvision is the label value of the data obtained by folder
                 # So the serial number here is actually the label of the classification.
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path


def load_query_strings(filename,
                       prepend_string = '',
                       tokenize=True,
                       device=__default_device):
    qs = []
    with open(filename, 'r') as f:
        for l in f:

            if len(prepend_string) > 0:
                _qs = f'{prepend_string} {l.strip()}'
            else:
                _qs = l.strip()

            qs.append(_qs)

    logging.debug(f"Loaded {len(qs)} queries")

    if tokenize:
        qs = clip.tokenize(qs).to(device)
        logging.debug(f"Tokenized to {qs.shape} tensor")

    return qs

def main(argv):
    if argv:
        print('Unprocessed args:', argv)

    model, preprocess = clip.load(FLAGS.clip_model, FLAGS.device)
    qs = load_query_strings(FLAGS.queries, FLAGS.prepend)
    with torch.no_grad():
        qs_embeddings = model.encode_text(qs)
        qs_embeddings /= qs_embeddings.norm(dim=-1, keepdim=True)
        logging.debug(f'Text embedding tensor size: {qs_embeddings.shape}')

    image_dataset = ImageFolderFilename(FLAGS.imagepath, transform=preprocess)
    logging.debug(f'{len(image_dataset)} images found')
    image_dataloader = DataLoader(image_dataset, batch_size=FLAGS.batchsize)

    df = pd.DataFrame()

    for i in tqdm(image_dataloader):
        _df = pd.DataFrame()

        _df['fn']=i[1]


        with torch.no_grad():
            img_embeddings = model.encode_image(i[0])
            img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
    
            logging.debug(f'Batch tensor size: {img_embeddings.shape}')
    
            cos_distance = img_embeddings @ qs_embeddings.T
            logging.debug(f'Batch distance tensor size: {cos_distance.shape}')

            feat_dim = list(range(cos_distance.shape[-1]))
            _df[feat_dim] = np.array(cos_distance.cpu())

    
            df = df.append(_df)

    df.to_csv(FLAGS.outfile, index=False)



if __name__ == '__main__':
    app.run(main)
