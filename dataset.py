import glob
import os
import numpy as np
import cv2

from torch.utils.data import Dataset 
from PIL import Image
import torchvision.transforms as transforms

class SegDataset(Dataset):
    def __init__(self, dataRoot, transforms_= None, transforms_mask = None, subFold="Train_NG", isTrain=True):

        self.isTrain = isTrain
        if transforms_mask == None:
            self.maskTransform = transforms.Compose([transforms.ToTensor()])
        else:
            self.maskTransform = transforms_mask

        if transforms_== None:
            self.transform = self.maskTransform
        else:
            self.transform = transforms_

        self.imgFiles   = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.jpg"))

        if isTrain:
            self.labelFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.bmp"))

        self.len = len(self.imgFiles)

    def __getitem__(self, index):
        
        idx = index %  self.len

        img  = Image.open(self.imgFiles[idx]).convert("RGB")
        img = self.transform(img)
        
        if self.isTrain==True:
            #mask = Image.open(self.labelFiles[idx]).convert("RGB")   
            mat = cv2.imread(self.labelFiles[idx], cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5, 5), np.uint8)
            matD = cv2.dilate(mat, kernel)
            mask = Image.fromarray(matD)               # image2 is a PIL image    
            mask = self.maskTransform(mask)
            return {"img":img, "mask":mask}
        else:
            return {"img":img}

    def __len__(self):
        return len(self.imgFiles)
