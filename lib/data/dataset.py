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
  def __init__(self, root, dataframe=None, csv_file=None, transform=None):
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
    self.root = root

  def __len__(self):
    return len(self.image_df)

  def __getitem__(self, idx):

    rows = self.image_df.iloc[idx]
    groundtruth = pil_loader(os.path.join(self.root, rows['groundtruth_source']))
    mask = pil_loader(os.path.join(self.root, rows['mask_source']))
    segment = np.load(os.path.join(self.root, rows['segment']))
    segment = torch.from_numpy(segment)
    try:
      damage_type = rows['damage_type']
    except:
      damage_type = ''
    if self.transform:
      groundtruth = self.transform(groundtruth)
      mask = self.transform(mask)
      prefilled = self.transform(prefilled)

    return groundtruth,mask,segment