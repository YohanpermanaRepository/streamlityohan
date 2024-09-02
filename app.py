import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt
from io import BytesIO
import glob
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
# Load model

def upsample(c_in, c_out, dropout=False):
  result = nn.Sequential()
  result.add_module('con', nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False))
  result.add_module('bat',nn.BatchNorm2d(c_out))
  if dropout:
    result.add_module('drop',nn.Dropout2d(0.5, inplace=True))
  result.add_module('relu',nn.ReLU(inplace=False))

  return result

def downsample(c_in, c_out, batchnorm=True):
  result = nn.Sequential()
  result.add_module('con', nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False))

  if batchnorm:
    result.add_module('batc',nn.BatchNorm2d(c_out))
  result.add_module('LRelu',nn.LeakyReLU(0.2, inplace=False))

  return result


class Generator(nn.Module):
  def __init__(self, input_nc=1, output_nc=2, n_filters=64):
    super(Generator, self).__init__()


    layer1 = nn.Conv2d(input_nc, n_filters, kernel_size=4, stride=2, padding=1, bias=False)
    layer2 = downsample(n_filters,n_filters*2)
    layer3 = downsample(n_filters*2, n_filters*4)
    layer4 = downsample(n_filters*4, n_filters*8)
    layer5 = downsample(n_filters*8, n_filters*8)
    layer6 = downsample(n_filters*8, n_filters*8)
    layer7 = downsample(n_filters*8, n_filters*8)
    layer8 = downsample(n_filters*8, n_filters*8)

    #decoder
    d_inc = n_filters*8
    dlayer8 = upsample(d_inc, n_filters*8, dropout=True)
    d_inc = n_filters*8*2
    dlayer7 = upsample(d_inc, n_filters*8, dropout=True)
    d_inc = n_filters*8*2
    dlayer6 = upsample(d_inc, n_filters*8, dropout=True)
    d_inc = n_filters*8*2
    dlayer5 = upsample(d_inc, n_filters*8)
    d_inc = n_filters*8*2
    dlayer4 = upsample(d_inc, n_filters*4)
    d_inc = n_filters*4*2
    dlayer3 = upsample(d_inc, n_filters*2)
    d_inc = n_filters*2*2
    dlayer2 = upsample(d_inc, n_filters)

    dlayer1 = nn.Sequential()
    d_inc = n_filters*2
    dlayer1.add_module('relu', nn.ReLU(inplace=False))
    dlayer1.add_module('t_conv', nn.ConvTranspose2d(d_inc, output_nc, kernel_size=4, stride=2, padding=1, bias=False))
    dlayer1.add_module('tanh', nn.Tanh())

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1


  def forward(self, input):
    out1 = self.layer1(input)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    return dout1


# Fungsi untuk mengonversi Lab ke RGB
def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

import gdown
import os
import torch
import torch.nn as nn

# Fungsi unduhan file model
def download_model_if_not_exists(model_path, file_id):
    if not os.path.exists(model_path):
        download_url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(download_url, model_path, quiet=False)

# Tentukan path untuk menyimpan model
model_path = 'model.pth'
file_id = '1PMQVxvDTmLqP1DhX8xmP3K_iw_RCJsQN'

# Unduh model jika belum ada
download_model_if_not_exists(model_path, file_id)

# Load model dan mengatur ke mode evaluasi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_G = Generator().to(device)
net_G.load_state_dict(torch.load(model_path, map_location=device))
net_G.eval()

# Aplikasi Streamlit
st.title('Generative Adversarial Network Coloring Batik')

# Pengunggah file (dengan multiple file upload)
uploaded_files = st.file_uploader("Choose images...", type="jpg", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Crop gambar agar ukuran sama
        size = (256, 256)  # Ukuran yang diinginkan
        image_cropped = image.resize(size, Image.LANCZOS)  # Menggunakan LANCZOS sebagai alternatif

        # Pra-pemrosesan gambar
        img = np.array(image_cropped)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1.  # Saluran warna luminance

        # Membuat tensor
        L = L.unsqueeze(0).to(device)

        # Membuat Gambar Grayscale dari saluran L
        gray_image = (L.squeeze().cpu().numpy() + 1.) * 255 / 2  # Mengonversi saluran L ke rentang [0, 255]
        gray_image = gray_image.astype(np.uint8)  # Mengubah ke uint8

        # Meneruskan melalui model
        with torch.no_grad():
            fake_color = net_G(L)
            fake_color = fake_color.detach()

        # Mengonversi Lab ke RGB
        fake_imgs = lab_to_rgb(L, fake_color)
        fake_img = fake_imgs[0]

        # Menampilkan gambar keluaran dalam satu baris
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image_cropped, caption='Uploaded Image', use_column_width=True)

        with col2:
            st.image(gray_image, caption='Grayscale Image (L channel)', use_column_width=True, clamp=True)

        with col3:
            st.image(fake_img, caption='Colorized Image', use_column_width=True)

        # Opsi untuk mengunduh hasil
        result = Image.fromarray((fake_img * 255).astype(np.uint8))
        buf = BytesIO()
        result.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(f"Download Result for {uploaded_file.name}", data=byte_im, file_name=f"colorized_image_{uploaded_file.name}", mime="image/jpeg")



