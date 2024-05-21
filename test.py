import os
import sys, time
import tifffile as tiff
from tqdm import tqdm
from PIL import Image
from timm.models.layers import trunc_normal_

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms.autoaugment import InterpolationMode
from torchvision import transforms
from torchvision.utils import save_image
from segmentation_models_pytorch.losses import FocalLoss
from torchmetrics.classification import  BinarySpecificity, BinaryRecall, BinaryF1Score, BinaryAUROC, BinaryAccuracy

# Seed everything
random_seed = 12345
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# Defining a custom Dataset class
class DRIVE(Dataset):
    def __init__(self, input_dir, output_dir, transform=None, train=False):
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.transform  = transform
        self.train = train
        self.images = sorted(os.listdir(input_dir))
        self.masks = sorted(os.listdir(output_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path    = os.path.join(self.input_dir, self.images[index])
        # img_id = img_path.split('/')[-1]
        # mask_path   = os.path.join(self.output_dir, img_id)
        mask_path = os.path.join(self.output_dir, self.masks[index])
        if self.train:
            # img = torch.from_numpy(np.load(img_path))
            # mask = torch.from_numpy(np.load(mask_path))
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            if self.transform is not None:
                img   = self.transform['train']['input'](img)
                # print(img.dtype)  # float32
                # img = img.to(dtype=torch.float16)
                # img = img[:,:,:-1]
                img /= 255

                mask = self.transform['train']['mask'](mask)
                # print(mask.dtype)   # float32
                # mask = mask.to(dtype=torch.float16).to(dtype=torch.uint8)
                if mask.max() > 0:
                    mask /= mask.max()
        else:
            img         = tiff.imread(img_path)
            mask        = Image.open(mask_path)
            mask.convert('L')
            if self.transform is not None:
                img = Image.fromarray(img)
                img   = self.transform['test']['input'](img)
                # img = img[:,:,:-1]
                img /= 255
                # mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = self.transform['test']['mask'](mask)
                if mask.max() > 0:
                    mask /= mask.max()
        
        # mask = torch.where(mask>0, 1, 0)
        # mask = mask.to(dtype=torch.uint8)
        return img, mask
    
train_input_dir = 'DRIVE_Aug_Data/train/images'
train_output_dir = 'DRIVE_Aug_Data/train/masks'

test_input_dir = 'DRIVE--Digital-Retinal-Images-for-Vessel-Extraction/DRIVE/test/images'
test_output_dir = 'DRIVE--Digital-Retinal-Images-for-Vessel-Extraction/DRIVE/test/1st_manual'

transform = {
    'train':{
        'input': transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ]),
    'mask': transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    },
    'test':{
        'input': transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ]),
    'mask': transforms.Compose([
        transforms.Resize((512,512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])}  
}

# Creating Dataset and Dataloader objects
train_data = DRIVE(train_input_dir, train_output_dir, transform, train=True)
print(len(train_data))
train_data, val_data = random_split(train_data, [0.85, 0.15], generator=torch.Generator().manual_seed(random_seed))
test_data = DRIVE(test_input_dir, test_output_dir, transform)
print('Training samples:',len(train_data))
print('Validation samples:',len(val_data))
print('Test samples:',len(test_data))

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

class FeatureAggregationModule(nn.Module):
   def __init__(self, inch,outch):
     super(FeatureAggregationModule,self).__init__()
     self.conv1 = nn.Conv2d(inch, outch, kernel_size=1, padding =0, bias=False)
     self.conv2 = nn.Conv2d(inch, outch, kernel_size=3, padding =1, bias=False)
     self.conv3 = nn.Conv2d(inch, outch, kernel_size=3, padding =2, dilation=2,bias=False)
     self.norm = nn.BatchNorm2d(outch)

   def forward(self, x):
     x1 = self.conv1(x)
     x2 = self.conv2(x)
     x3 = self.conv3(x)
     x = self.norm(x1+x2+x3)
     return x

class ModifiedResidualBlock(nn.Module):
    def __init__(self, inch, outch):
      super(ModifiedResidualBlock,self).__init__()
      self.conv1 = nn.Conv2d(inch, outch, kernel_size=3,padding=1,bias=False)
      self.batch = nn.BatchNorm2d(outch)
      self.drop1 = nn.Dropout2d(0.2)
      self.relu1 = nn.LeakyReLU(0.1,inplace=True)
      self.conv2 = nn.Conv2d(outch, outch, kernel_size=3,padding=1,bias=False)
      self.batch2 = nn.BatchNorm2d(outch)
      self.drop2 = nn.Dropout2d(0.2)
      self.leakyrelu = nn.LeakyReLU(0.1,inplace=True)
      if inch != outch:
            self.conv3 = nn.Conv2d(inch, outch, kernel_size=1, padding=0)
      else:
            self.conv3 = None

    def forward(self, x):
      xi = self.conv1(x)
      x1 = self.batch(xi)
      x1 = self.drop1(x1)
      x1 = self.relu1(x1)
      x1 = self.conv2(x1)
      x1 = self.batch2(x1)
      x1 = self.drop2(x1)
      if self.conv3 is not None:
          x = self.conv3(x)
      x1 = self.leakyrelu(x1+x)

      return x1

class Upsample(nn.Module):
  def __init__(self, inch, outch):
    super(Upsample,self).__init__()
    self.conv1 = nn.ConvTranspose2d(inch,outch,kernel_size =2,padding =0, stride =2, bias=False)
    self.batch = nn.BatchNorm2d(outch)
    self.relu = nn.LeakyReLU(0.1,inplace=True)

  def forward(self,x):
    x1 = self.conv1(x)
    x1 = self.batch(x1)
    x1 = self.relu(x1)

    return x1

class Downsample(nn.Module):
  def __init__(self, inch, outch):
    super(Downsample,self).__init__()
    self.conv1 = nn.Conv2d(inch,outch,kernel_size =2,padding =0, stride =2, bias=False)
    self.batch = nn.BatchNorm2d(outch)
    self.relu = nn.LeakyReLU(0.1,inplace=True)

  def forward(self,x):
    x1 = self.conv1(x)
    x1 = self.batch(x1)
    x1 = self.relu(x1)
    return x1

class DeepBlock(nn.Module):
    def __init__(self,inch, outch ):
        super(DeepBlock, self).__init__()
        self.fam =  FeatureAggregationModule(inch, outch)
        self.mod = ModifiedResidualBlock(outch, outch)

    def forward(self,x):
        x1 = self.fam(x)
        x1 = self.mod(x1)
        return x1

class SpatialAttn(nn.Module):
  def __init__(self):
    super(SpatialAttn, self).__init__()
    self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=(7-1)//2, bias=False)

  def forward(self, x):
    avg_pool_output = torch.mean(x, dim=1)
    max_pool_output, _ = torch.max(x, dim=1)
    concat_output = torch.cat((avg_pool_output.unsqueeze(1), max_pool_output.unsqueeze(1)), dim=1)
    output = self.conv(concat_output)
    output = torch.sigmoid(output)
    output_ftrs = x * output
    return output_ftrs

class FRUNet(nn.Module):
  def __init__(self):
    super(FRUNet, self).__init__()
    self.mod11 = ModifiedResidualBlock(1, 32)
    self.mod12 = ModifiedResidualBlock(32, 32)
    self.db11 = DeepBlock(32*2,32)
    self.db12  = DeepBlock(32*2 ,32)
    self.db13 = DeepBlock(32*2, 32)
    self.db14 = DeepBlock(32*2, 32)
    self.db15 = DeepBlock(32*2,32)
    self.final1 = nn.Conv2d(32, 1, kernel_size=1, stride=1,bias=True)
    self.final2 = nn.Conv2d(32, 1, kernel_size=1, stride=1,bias=True)
    self.final3 = nn.Conv2d(32, 1, kernel_size=1, stride=1,bias=True)
    self.final4 = nn.Conv2d(32, 1, kernel_size=1, stride=1,bias=True)
    self.final5 = nn.Conv2d(32, 1, kernel_size=1, stride=1,bias=True)
    self.down11 = Downsample(32, 64)
    self.up11 = Upsample(64, 32)
    self.down12 =Downsample(32, 64)
    self.up12 = Upsample(64, 32)
    self.down13 =Downsample(32, 64)
    self.up13 = Upsample(64, 32)
    self.down14 =Downsample(32, 64)
    self.up14 = Upsample(64, 32)
    self.down15 = Downsample(32, 64)
    self.up15 = Upsample(64, 32)
    self.layer21 = ModifiedResidualBlock(64, 64)
    self.db21 = DeepBlock(64*2,64)
    self.db22 = DeepBlock(64*3, 64)
    self.db23 = DeepBlock(64*3, 64)
    self.db24 = DeepBlock(64*3, 64)
    self.down21 = Downsample(64, 128)
    self.up21 = Upsample(128, 64)
    self.down22 =Downsample(64, 128)
    self.up22 = Upsample(128, 64)
    self.down23 =Downsample(64, 128)
    self.up23 = Upsample(128, 64)
    self.layer31 = ModifiedResidualBlock(128, 128)
    self.db31 = DeepBlock(128*2, 128)
    self.db32 = DeepBlock(128*3, 128)
    self.down31 = Downsample(128, 256)
    self.up31 = Upsample(256, 128)
    self.splattn = SpatialAttn()
    # self.apply(InitWeights_He)

  def forward(self,x):
    out11 = self.mod11(x)
    out12 = self.mod12(out11)
    out21 = self.layer21(self.down11(out11))
    out31 = self.layer31(self.down21(out21))
    out41 = self.down31(out31)
    out42 = self.splattn(out41)
    out43 = self.up31(out42)
    res1 = self.db11(torch.cat((out12,self.up11(out21)),dim=1))
    out22 = self.db21(torch.cat((self.down12(out12),out21),dim=1))
    res2 = self.db12(torch.cat((res1,self.up12(out22)),dim=1))
    out23 = self.db22(torch.cat((self.down13(res1),out22,self.up21(out31)),dim=1))
    res3 = self.db13(torch.cat((res2,self.up13(out23)),dim=1))
    out32 = self.db31(torch.cat((out31,self.down22(out22)),dim=1))
    out24 = self.db23(torch.cat((self.down14(res2),out23,self.up22(out32)),dim=1))
    res4 = self.db14(torch.cat((res3,self.up14(out24)),dim=1))
    # print(out43.shape)
    # out33 = self.db32(torch.cat((out32,self.down23(out23),torch.stack((out43,torch.ones(1,128,128,1).to(device)),3)),dim=1))
    out33 = self.db32(torch.cat((out32,self.down23(out23),out43), dim=1))
    out25 = self.db24(torch.cat((self.down15(res3),out24,self.up23(out33)),dim=1))
    res5 = self.db15(torch.cat((res4,self.up15(out25)),dim=1))

    res = 0.2*self.final1(res1) + 0.2*self.final2(res2) + 0.2*self.final3(res3) + 0.2*self.final4(res4) + 0.2*self.final5(res5)
    res = torch.sigmoid(res)
    return res
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = FRUNet()
net.to(device)
net.load_state_dict(torch.load('results/saved_models/FRUNet_SA_best.pth'))

# Initialize metrics
auroc = BinaryAUROC().to(device)
f1 = BinaryF1Score().to(device)
spec = BinarySpecificity().to(device)
sen = BinaryRecall().to(device)
acc = BinaryAccuracy().to(device)

# Test set analysis
test_spec = 0
test_sen = 0
test_aucroc = 0
test_f1 = 0
test_acc = 0

net.eval()
with torch.no_grad():
  t_acc, t_aucroc, t_f1, t_spec, t_sen = 0,0,0,0,0
  num_batches = 0

  for imgs, masks in (test_dataloader):
    imgs, masks = imgs.to(device), masks.to(device)
    outputs = net(imgs)
    outputs = (outputs>0.5).type(torch.int8)
    t_acc += acc(outputs, masks).item()
    t_f1 += f1(outputs, masks).item()
    t_spec += spec(outputs, masks).item()
    t_sen += sen(outputs, masks).item()
    t_aucroc += auroc(outputs, masks).item()
    num_batches+=1
    
  test_acc = t_acc/ num_batches
  test_spec = t_spec/num_batches
  test_sen = t_sen/num_batches
  test_f1 = t_f1/num_batches
  test_aucroc = t_aucroc/num_batches

print('---- Test Set Results ----')
print('Accuracy : {:.4f}'.format(test_acc))
print('Sensitivity : {:.4f}'.format(test_sen))
print('Specificity : {:.4f}'.format(test_spec))
print('F1 Score : {:.4f}'.format(test_f1))
print('AUCROC Score : {:.4f}'.format(test_aucroc))