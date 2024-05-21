import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import sys, time
import tifffile as tiff
from PIL import Image
from torchvision.transforms.autoaugment import InterpolationMode
from torchvision import transforms
from torchvision.utils import save_image

# Seed everything
random_seed = 12345
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class Fix_RandomRotation(object):
    def __init__(self, degrees=360, resample=False, expand=False, center=None):
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params():
        p = torch.rand(1)

        if p >= 0 and p < 0.33:
            angle = 180
        elif p >= 0.33 and p < 0.66:
            angle = 270
        else:
            angle = 90
        return angle

    def __call__(self, img, angle=None):
        if angle == None:
            angle = self.get_params()
        return transforms.functional.rotate(img, angle, self.resample, self.expand, self.center), angle

    def __repr__(self):
        format_string = self.__class__.__name__ + \
            '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
    
# define an augmentation function
def augmentData(image,mask,img_id=None, save_dir=None, plot=False):
  # takes a batch of image, labels and returns a new batch comprising of original, flipped horizontal, flipped vertical and rotated images, and correpsonding labels
  # image (N,C,H,W)
  # mask (N,H,W)
  flipped_h_img = transforms.RandomHorizontalFlip(1)(image)
  flipped_v_img = transforms.RandomVerticalFlip(1)(image)
  rotated_img, angle = Fix_RandomRotation()(image)
  x = torch.stack((image, flipped_h_img, flipped_v_img, rotated_img))

  flipped_h_mask = transforms.RandomHorizontalFlip(1)(mask)
  flipped_v_mask = transforms.RandomVerticalFlip(1)(mask)
  rotated_mask, _ = Fix_RandomRotation()(mask, angle)
  y = torch.stack((mask, flipped_h_mask, flipped_v_mask, rotated_mask))

  if plot:
    plt.figure(figsize=(6,10))
    for i, sample in enumerate(zip(x,y)):
      plt.subplot(4,2,2*i+1)
      plt.imshow(sample[0].numpy().squeeze(), cmap='gray')
      plt.subplot(4,2,2*i+2)
      plt.imshow(sample[1].numpy().squeeze(), cmap='gray')
    plt.show()

  if save_dir != None:
    save_images_dir = os.path.join(save_dir,'train','images')
    save_masks_dir = os.path.join(save_dir, 'train','masks')
    os.makedirs(save_images_dir, exist_ok=True)
    os.makedirs(save_masks_dir, exist_ok=True)

    image_name = [img_id+'_original', img_id+'_hflipped', img_id+'_vflipped', img_id+f'_rotated_{angle}']
    image_name = [name + '.png' for name in image_name]
    
    save_image(image, os.path.join(save_images_dir, image_name[0]))
    # mask = (mask/mask.max())
    save_image(mask, os.path.join(save_masks_dir, image_name[0]))
    # image, mask = image.numpy(), mask.numpy()
    # if not(np.any(np.isnan(image)) or np.any(np.isnan(mask))):
    #     np.save(os.path.join(save_images_dir, image_name[0]), image)
    #     np.save(os.path.join(save_masks_dir, image_name[0]), mask)
    # else:
    #     with open('error_samples.txt', 'a') as f:
    #         f.write(f'{image_name[0]}\n')
    print(f'{image_name[0]} done!')

    save_image(flipped_h_img, os.path.join(save_images_dir, image_name[1]))
    # flipped_h_mask = (flipped_h_mask/flipped_h_mask.max())
    save_image(flipped_h_mask, os.path.join(save_masks_dir, image_name[1]))
    # flipped_h_img, flipped_h_mask = flipped_h_img.numpy(), flipped_h_mask.numpy()
    # if not(np.any(np.isnan(flipped_h_img)) or np.any(np.isnan(flipped_h_mask))):
    #     np.save(os.path.join(save_images_dir, image_name[1]), flipped_h_img)
    #     np.save(os.path.join(save_masks_dir, image_name[1]), flipped_h_mask)
    # else:
    #     with open('error_samples.txt', 'a') as f:
    #         f.write(f'{image_name[1]}\n')
    print(f'{image_name[1]} done!')

    save_image(flipped_v_img, os.path.join(save_images_dir, image_name[2]))
    # flipped_v_mask = (flipped_v_mask/flipped_v_mask.max())
    save_image(flipped_v_mask, os.path.join(save_masks_dir, image_name[2]))
    # flipped_v_img, flipped_v_mask = flipped_v_img.numpy(), flipped_v_mask.numpy()
    # if not(np.any(np.isnan(flipped_v_img)) or np.any(np.isnan(flipped_v_mask))):
    #     np.save(os.path.join(save_images_dir, image_name[2]), flipped_v_img)
    #     np.save(os.path.join(save_masks_dir, image_name[2]), flipped_v_mask)
    # else:
    #     with open('error_samples.txt', 'a') as f:
    #         f.write(f'{image_name[2]}\n')
    print(f'{image_name[2]} done!')

    save_image(rotated_img, os.path.join(save_images_dir, image_name[3]))
    # rotated_mask = (rotated_mask/rotated_mask.max())
    save_image(rotated_mask, os.path.join(save_masks_dir, image_name[3]))
    # rotated_img, rotated_mask = rotated_img.numpy(), rotated_mask.numpy()
    # if not(np.any(np.isnan(rotated_img)) or np.any(np.isnan(rotated_mask))):
    #     np.save(os.path.join(save_images_dir, image_name[3]), rotated_img)
    #     np.save(os.path.join(save_masks_dir, image_name[3]), rotated_mask)
    # else:
    #     with open('error_samples.txt', 'a') as f:
    #         f.write(f'{image_name[3]}\n')
    print(f'{image_name[3]} done!')        
        
  return x,y, image_name

def create_patches(images, kernel=96, stride=16, image_name=None, save_dir=None):
    # image (N,C,H,W)
    shape = (images.shape[-2],images.shape[-1])
    for img, img_name in zip(images, image_name):
        # print(img.shape)    # (C,H,W)
        pad_h = stride - (shape[0] - kernel) % stride
        pad_w = stride - (shape[1] - kernel) % stride
        image = F.pad(img, (0, pad_w, 0, pad_h), "constant", 0)
        image = image.unfold(1, kernel, stride).unfold(2, kernel, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(image.shape[0] * image.shape[1], image.shape[2], kernel, kernel)
        for i, patch in enumerate(image):
            # print(patch.shape)  # (C,k,k)
            patch = transforms.Resize(shape,interpolation=InterpolationMode.NEAREST)(patch)
            save_path = os.path.join(save_dir, f'{img_name[:-4]}_patch_{i}.png')
            # if 'masks' in save_dir:
            #    patch = (patch/patch.max())
            save_image(patch, save_path)
            # patch = patch.numpy()
            # if not(np.any(np.isnan(patch))):
            #     np.save(save_path, patch)
            # else:
            #     with open('error_samples.txt', 'a') as f:
            #         f.write(f'{img_name}_patch_{i}\n')
            print(f'{img_name[:-4]}_patch_{i} done!')

transform = {
    'input': transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ]),
    'mask': transforms.Compose([
        transforms.Resize((512,512), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
}

if __name__=='__main__':

    train_images_dir = 'DRIVE--Digital-Retinal-Images-for-Vessel-Extraction/DRIVE/training/images'
    train_masks_dir = 'DRIVE--Digital-Retinal-Images-for-Vessel-Extraction/DRIVE/training/1st_manual'

    images = sorted(os.listdir(train_images_dir))
    masks = sorted(os.listdir(train_masks_dir))

    for i, sample in enumerate(zip(images,masks)):
        image, mask = sample[0], sample[1]
        img_id = image[:2]
        img_pth = os.path.join(train_images_dir,image)
        mask_pth = os.path.join(train_masks_dir, mask)
        print(f"Processing image {img_id}...")

        img = tiff.imread(img_pth)
        img = Image.fromarray(img)
        mask = Image.open(mask_pth)

        img, mask = transform['input'](img), transform['mask'](mask)
        # print(img.shape)    # (C,H,W)
        # print(mask.shape)   # (C,H,W)

        aug_imgs, aug_masks, image_name = augmentData(img, mask, img_id, save_dir='DRIVE_Aug_Data')
        create_patches(aug_imgs, image_name=image_name, save_dir='DRIVE_Aug_Data/train/images')
        create_patches(aug_masks, image_name=image_name, save_dir='DRIVE_Aug_Data/train/masks')
        print(f"Image {img_id} done!\n") 