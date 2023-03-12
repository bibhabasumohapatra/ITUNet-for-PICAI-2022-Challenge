
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from model import itunet_2d


import os 
from tqdm import tqdm


## CONFIG:
tumor = True

if tumor:
   IMG_PATH = ""
else:
   IMG_PATH = ""


class Normalize(object):
  def __call__(self,sample):
    ct = sample['ct']
    seg = sample['seg']
    for i in range(ct.shape[0]):
        if np.max(ct[i])!=0:
            ct[i] = ct[i]/np.max(ct[i])
        
    ct[ct<0] = 0

    new_sample = {'ct':ct, 'seg':seg}
    return new_sample
  
class To_Tensor(object):
  '''
  Convert the data in sample to torch Tensor.
  Args:
  - n_class: the number of class
  '''
  def __init__(self,num_class=2,input_channel = 3):
    self.num_class = num_class
    self.channel = input_channel

  def __call__(self,sample):

    ct = sample['ct']
    seg = sample['seg']

    new_image = ct[:self.channel,...]
    new_label = np.empty((self.num_class,) + seg.shape, dtype=np.float32)
    for z in range(1, self.num_class):
        temp = (seg==z).astype(np.float32)
        new_label[z,...] = temp
    new_label[0,...] = np.amax(new_label[1:,...],axis=0) == 0   
   
    # convert to Tensor
    new_sample = {'image': torch.from_numpy(new_image),
                  'label': torch.from_numpy(new_label)}
    
    return new_sample

class DataGenerator(Dataset):
  '''
  Custom Dataset class for data loader.
  Args：
  - path_list: list of file path
  - roi_number: integer or None, to extract the corresponding label
  - num_class: the number of classes of the label
  - transform: the data augmentation methods
  '''
  def __init__(self, path_list, num_class=2, transform=None):

    self.path_list = path_list
    self.num_class = num_class
    self.transform = transform


  def __len__(self):
    return len(self.path_list)

  def __getitem__(self,index):

    # ct = hdf5_reader(self.path_list[index],'ct')
    # seg = hdf5_reader(self.path_list[index],'seg') 
    
    ct = np.load(f"{IMG_PATH}/{self.path_list[index]}.npy").astype(np.float32)
    seg = np.load(f"/mnt/prj001/Bibhabasu_Mohapatra/slices_masksv4/{self.path_list[index]}").astype(np.float32)

    sample = {'ct': ct, 'seg':seg}
    # Transform

    if self.transform is not None:
      sample = self.transform(sample)

    return sample


## ENGINE 

@torch.inference_mode()
def inference(data_loader, model, device="cuda"):
    model.eval()
    final_output = []

    tk0 = tqdm(data_loader,colour='#5d3fd3')

    for batch in tk0:
        image = batch["image"]

        image = image.to(device, dtype=torch.float)
        
        output = model(image)[0]

        output = torch.softmax(output, dim=1)
        output = torch.argmax(output,dim=1)
        
        final_output.append(output.detach().cpu().numpy())
    tk0.close()
    return final_output

## MAIN

if not tumor:
   image_ids = os.listdir(IMG_PATH)


MODEL_WEIGHT = "/home/bibhabasum/ITUNet-for-PICAI-2022-Challenge/segmentation/new_ckpt/seg/itunet_d24/fold1/epoch:154-train_loss:1139.46366-train_dice:0.89235-train_run_dice:0.81933-val_loss:2130.94495-val_dice:0.88348-val_run_dice:0.75310.pth"

model = itunet_2d(n_channels=3,n_classes=2, image_size= tuple((384,384)), transformer_depth = 24)
checkpoint = torch.load(MODEL_WEIGHT)
model.load_state_dict(checkpoint['state_dict'])

model = model.cuda()

inference_transforms = transforms.Compose([
            Normalize(), 
            To_Tensor(num_class=2,input_channel=3)
        ])

dataset = DataGenerator(image_ids,transform=inference_transforms)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
outputs = inference(data_loader=dataloader,model=model)

from skimage import filters
for i in range(len(outputs)):
    mask = outputs[i][0]

    np.save(f"/home/bibhabasum/pred_dump/{image_ids[i]}",mask)
