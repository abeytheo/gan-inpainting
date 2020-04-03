from PIL import Image
import pandas as pd
import torch
import os

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)

        ### convert to greyscale
        return img.convert('L')

class InpaintingDataset(torch.utils.data.Dataset):
  def __init__(self, dataframe=None, csv_file=None, transform=None):
    """
      dataframe: A pandas dataframe
      csv_file: A csv file name
      transform: Tensor transformation list to be applied
    
      The dataframe or csv file needs to supply two fields, groundtruth_source and mask_source.
    """
    if dataframe is not None:
      self.image_df = dataframe
    elif csv_file:
      self.image_df = pd.read_csv(csv_file)
    else:
      raise Exception("Please supply dataframe or file path")
    self.transform = transform

  def __len__(self):
    return len(self.image_df)

  def __getitem__(self, idx):

    rows = self.image_df.iloc[idx]
    groundtruth = pil_loader(os.path.join(rows['groundtruth_source']))
    mask = pil_loader(os.path.join(rows['mask_source']))
    try:
      damage_type = rows['damage_type']
    except:
      damage_type = ''
    if self.transform:
      groundtruth = self.transform(groundtruth)
      mask = self.transform(mask)

    return groundtruth,mask,damage_type