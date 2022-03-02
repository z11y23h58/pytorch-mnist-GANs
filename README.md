# pytorch-mnist-GANs


## 读取数据

~~~python
import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

image_size = 64
batch_size = 128

# mnist图片为28*28，放大到64*64，并归一化到[-1, 1]
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)
])

# 在当前目录下创建data文件夹，放入MNIST数据集
mnist = torchvision.datasets.MNIST("./data", train = True, transform = transform, download = True)
data_loader = DataLoader(dataset = mnist, batch_size = batch_size, shuffle = True)

# 创建samples_GAN文件夹
import os
samples_dir = "samples_GAN"
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

# 保存图片到samples_dir
from torchvision.utils import save_image

for images, _ in data_loader:
    break
save_image(images, os.path.join(samples_dir, "real_images.png"))
~~~


## GAN

## DCGAN

## WGAN

## CGAN
