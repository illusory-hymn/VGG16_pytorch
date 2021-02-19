import torch.nn as nn
import torch 
from torch.autograd import Variable
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os


class VGG16(nn.Module):
    def __init__(self):  ##  输入224x224
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

##  导入权重文件
vgg16 = VGG16().cuda()
vgg16.eval()
pretrain_path = 'vgg16-397923af.pth'
checkpoint = torch.load(pretrain_path)
vgg16.load_state_dict(checkpoint)

##   图片处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
test_img = Image.open("0.jpg")
test_img = transform(test_img)
test_img = test_img.unsqueeze(0)
test_img = Variable(test_img).cuda()

##  预测
pred = vgg16(test_img)
idx = torch.max(pred, 1)[1].squeeze(0).cpu().numpy()

##  读入label.txt
txt_path = 'label.txt'
labels = []
with open(txt_path, 'r', encoding='utf-8') as f: ## 因为要读入中文，所以要加上encoding='utf-8'
    for lines in f:
        labels.append(lines[:-1])   
print("class:{}".format(idx))
print("name:{}".format(labels[idx]))

