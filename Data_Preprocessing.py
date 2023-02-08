import torch
import numpy as np
import pandas as pd
from PIL import Image

def LoadInputImages(pth):
    '''
    The LoadInputImages method is a support method that helps PreprocessBatch function to load images for inputting to the ML model.

    '''
    temp_inp = torch.tensor(np.array(Image.open(pth+'/'+str(0)+'.jpg'))).unsqueeze(0)
    for img_num in range(1,10,1):
        temp_inp = torch.cat((temp_inp,torch.tensor(np.array(Image.open(pth+'/'+str(img_num)+'.jpg'))).unsqueeze(0)),0)
    return temp_inp
 
def PreprocessBatch(batch):
    '''
    The PreprocessBatch function fetahces the images from the folder and gets them ready to input them into the CNN-LSTM model.

    '''
    batch_size = len(batch)
    out_batch_imgs = torch.zeros((batch_size,10,110,110))
    out_batch_y = torch.zeros(batch_size)
    for pth_num in range(batch_size):
        out_batch_imgs[pth_num] = LoadInputImages(batch[pth_num][0])
        out_batch_y[pth_num] = batch[pth_num][1]
    return out_batch_imgs, out_batch_y