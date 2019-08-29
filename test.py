from models import SegmentNet, DecisionNet, weights_init_normal
from dataset import KolektorDataset

import torch.nn as nn
import torch

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import sys
import argparse
import time
import PIL.Image as Image

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
parser.add_argument("--test_seg_epoch", type=int, default=60, help="test segment epoch")
parser.add_argument("--test_dec_epoch", type=int, default=60, help="test segment epoch")
parser.add_argument("--img_height", type=int, default=704, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")

opt = parser.parse_args()

print(opt)

dataSetRoot = "/home/sean/Projects/SegDecNet/Data"

# ***********************************************************************

# Build nets
segment_net = SegmentNet(init_weights=True)
decision_net = DecisionNet(init_weights=True)

if opt.cuda:
    segment_net = segment_net.cuda()
    decision_net = decision_net.cuda()

if opt.test_seg_epoch != 0:
    # Load pretrained models
    segment_net.load_state_dict(torch.load("./saved_models/segment_net_%d.pth" % (opt.test_seg_epoch)))

if opt.test_dec_epoch != 0:
    # Load pretrained models
    decision_net.load_state_dict(torch.load("./saved_models/decision_net_%d.pth" % (opt.test_dec_epoch)))

transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


testloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= None,  subFold="Test", isTrain=False),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

#segment_net.eval()
#decision_net.eval()

for i, testBatch in enumerate(testloader):
    
    torch.cuda.synchronize()

    t1 = time.time()
    imgTest = testBatch["img"].cuda()

    

    with torch.no_grad():
        rstTest = segment_net(imgTest)

    fTest = rstTest["f"]
    segTest = rstTest["seg"]

    with torch.no_grad():
        cTest = decision_net(fTest, segTest)

    torch.cuda.synchronize()
    t2 = time.time()

    if cTest.item() > 0.5:
        labelStr = "NG"
    else: 
        labelStr = "OK"

    save_path_str = os.path.join(dataSetRoot, "testResult")

    if os.path.exists(save_path_str) == False:
        os.makedirs(save_path_str, exist_ok=True)

    print("processing image NO %d, time comsuption %fs"%(i, t2 - t1))
    save_image(imgTest.data, "%s/img_%d_%s.jpg"% (save_path_str, i, labelStr))
    save_image(segTest.data, "%s/img_%d_seg_%s.jpg"% (save_path_str, i, labelStr))
