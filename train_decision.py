
from models import SegmentNet, DecisionNet, weights_init_normal
from dataset import KolektorDataset
import numpy as np

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
parser.add_argument("--gpu_num", type=int, default=1, help="number of gpu")
parser.add_argument("--worker_num", type=int, default=4, help="number of input workers")
parser.add_argument("--batch_size", type=int, default=4, help="batch size of input")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
parser.add_argument("--end_epoch", type=int, default=61, help="end_epoch")
parser.add_argument("--seg_epoch", type=int, default=50, help="pretrained segment epoch")

parser.add_argument("--need_test", type=bool, default=True, help="need to test")
parser.add_argument("--test_interval", type=int, default=10, help="interval of test")
parser.add_argument("--need_save", type=bool, default=True, help="need to save")
parser.add_argument("--save_interval", type=int, default=10, help="interval of save weights")


parser.add_argument("--img_height", type=int, default=704, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")


opt = parser.parse_args()

print(opt)

dataSetRoot = "./Data" # "/home/sean/Data/KolektorSDD_sean"

# ***********************************************************************

# Build nets
segment_net = SegmentNet(init_weights=True)
decision_net = DecisionNet(init_weights=True)

# Loss functions
#criterion_segment  = torch.nn.MSELoss()
criterion_decision = torch.nn.MSELoss()

if opt.cuda:
    segment_net = segment_net.cuda()
    decision_net = decision_net.cuda()
    #criterion_segment.cuda()
    criterion_decision.cuda()

if opt.gpu_num > 1:
    segment_net = torch.nn.DataParallel(segment_net, device_ids=list(range(opt.gpu_num)))
    decision_net = torch.nn.DataParallel(decision_net, device_ids=list(range(opt.gpu_num)))

if opt.begin_epoch != 0:
    # Load pretrained models
    decision_net.load_state_dict(torch.load("./saved_models/decision_net_%d.pth" % (opt.begin_epoch)))
else:
    # Initialize weights
    decision_net.apply(weights_init_normal)

# load pretrained segment parameters
segment_net.load_state_dict(torch.load("./saved_models/segment_net_%d.pth" % (opt.seg_epoch)))
#segment_net.eval()

# Optimizers
optimizer_dec = torch.optim.Adam(decision_net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_mask = transforms.Compose([
    transforms.Resize((opt.img_height//8, opt.img_width//8)),
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainOKloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask, subFold="Train_OK", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)

trainNGloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_,  transforms_mask= transforms_mask, subFold="Train_NG", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)
'''
trainloader =  DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_,  transforms_mask= transforms_mask, subFold="Train_ALL", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)
'''
testloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask= transforms_mask,  subFold="Test", isTrain=False),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)


for epoch in range(opt.begin_epoch, opt.end_epoch):
    
    iterOK = trainOKloader.__iter__()
    iterNG = trainNGloader.__iter__()

    lenNum = min( len(trainNGloader), len(trainOKloader))
    lenNum = 2*(lenNum-1)

    #decision_net.train()
    #segment_net.eval()
    # train *****************************************************************
    for i in range(0, lenNum):

        if i % 2 == 0:
            batchData = iterOK.__next__()
            #idx, batchData = enumerate(trainOKloader)
            gt_c = Variable(torch.Tensor(np.zeros((batchData["img"].size(0), 1))), requires_grad=False)
        else :
            batchData = iterNG.__next__()
            gt_c = Variable(torch.Tensor(np.ones((batchData["img"].size(0), 1))), requires_grad=False)
            #idx, batchData = enumerate(trainNGloader)

        if opt.cuda:
            img = batchData["img"].cuda()
            mask = batchData["mask"].cuda()
            gt_c = gt_c.cuda()
        else:
            img = batchData["img"]
            mask = batchData["mask"]

        rst = segment_net(img)

        f = rst["f"]
        seg = rst["seg"]

        optimizer_dec.zero_grad()

        rst_d = decision_net(f, seg)
        # rst_d = torch.Tensor.long(rst_d)

        loss_dec = criterion_decision(rst_d, gt_c)

        loss_dec.backward()
        optimizer_dec.step()

        sys.stdout.write(
            "\r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"
             %(
                epoch,
                opt.end_epoch,
                i,
                lenNum,
                loss_dec.item()
             )
        )
    
    # test ****************************************************************************
    if opt.need_test and epoch % opt.test_interval == 0 and epoch >= opt.test_interval:
        #decision_net.eval()
        #segment_net.eval()

        for i, testBatch in enumerate(testloader):
            imgTest = testBatch["img"].cuda()
            t1 = time.time()
            rstTest = segment_net(imgTest)           
            
            fTest = rstTest["f"]
            segTest = rstTest["seg"]

            cTest = decision_net(fTest, segTest)

            t2 = time.time()
            save_path_str = "./testResultDec/epoch_%d"%epoch
            if os.path.exists(save_path_str) == False:
                os.makedirs(save_path_str, exist_ok=True)
                #os.mkdir(save_path_str)

            if cTest.item() > 0.5:
                labelStr = "NG"
            else: 
                labelStr = "OK"

            print("processing image NO %d, time comsuption %fs"%(i, t2 - t1))
            save_image(imgTest.data, "%s/img_%d_%s.jpg"% (save_path_str, i , labelStr))
            save_image(segTest.data, "%s/img_%d_seg_%s.jpg"% (save_path_str, i, labelStr))
        
        #decision_net.train()

    # save parameters *****************************************************************
    if opt.need_save and epoch % opt.save_interval == 0 and epoch >= opt.save_interval:
        #decision_net.eval()

        save_path_str = "./saved_models"
        if os.path.exists(save_path_str) == False:
            os.makedirs(save_path_str, exist_ok=True)

        torch.save(decision_net.state_dict(), "%s/decision_net_%d.pth" % (save_path_str, epoch))
        print("save weights ! epoch = %d"%epoch)
        #decision_net.train()
        pass
