import torch
import torch.nn as nn

pi = 3.141592653589793

class RGB_HVI(nn.Module):
    def __init__(self):
        super(RGB_HVI, self).__init__()
        self.density_k = torch.nn.Parameter(torch.full([1],0.2)) # k is reciprocal to the paper mentioned
        self.gated = False
        self.gated2= False
        self.alpha = 1.0
        self.alpha_s = 1.3
        self.this_k = 0
        
    def HVIT(self, img):
        eps = 1e-8
        device = img.device
        dtypes = img.dtype
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(device).to(dtypes)
        value = img.max(1)[0].to(dtypes)
        img_min = img.min(1)[0].to(dtypes)
        hue[img[:,2]==value] = 4.0 + ( (img[:,0]-img[:,1]) / (value - img_min + eps)) [img[:,2]==value]
        hue[img[:,1]==value] = 2.0 + ( (img[:,2]-img[:,0]) / (value - img_min + eps)) [img[:,1]==value]
        hue[img[:,0]==value] = (0.0 + ((img[:,1]-img[:,2]) / (value - img_min + eps)) [img[:,0]==value]) % 6

        hue[img.min(1)[0]==value] = 0.0
        hue = hue/6.0

        saturation = (value - img_min ) / (value + eps )
        saturation[value==0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        
        k = self.density_k
        self.this_k = k.item()
        
        color_sensitive = ((value * 0.5 * pi).sin() + eps).pow(k)
        ch = (2.0 * pi * hue).cos()
        cv = (2.0 * pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I],dim=1)
        return xyz
    
    def PHVIT(self, img):
        eps = 1e-8
        H,V,I = img[:,0,:,:],img[:,1,:,:],img[:,2,:,:]
        
        # clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)
        
        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * pi).sin() + eps).pow(k)
        H = (H) / (color_sensitive + eps)
        V = (V) / (color_sensitive + eps)
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        h = torch.atan2(V + eps,H + eps) / (2*pi)
        h = h%1
        s = torch.sqrt(H**2 + V**2 + eps)
        
        if self.gated:
            s = s * self.alpha_s
        
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
                
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

def tensor_to_colormap(tensor, cmap='viridis'):
    """
    Map a single-channel tensor to a pseudo-color image.
    """
    tensor = tensor.squeeze().cpu().numpy()
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    colormap = plt.get_cmap(cmap)
    color_mapped = (colormap(tensor)[:, :, :3] * 255).astype('uint8')
    return color_mapped


def two_channel_to_rgb(t1, t2):
    """
    Map two channels to a pseudo-RGB image: H → R, V → G, and set B to zero.
    """
    t1 = t1.squeeze().cpu().numpy()
    t2 = t2.squeeze().cpu().numpy()

    # Normalize
    t1 = (t1 - t1.min()) / (t1.max() - t1.min() + 1e-8)
    t2 = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)

    H, W = t1.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :, 0] = (t1 * 255).astype(np.uint8)  # H -> R
    rgb[:, :, 1] = (t2 * 255).astype(np.uint8)  # V -> G
    # Set the B channel to zero.
    return rgb

def visualize_HVI(image_path, save_dir='/media/HDD/lvyou/HVI-CIDNet/output_vis'):
    os.makedirs(save_dir, exist_ok=True)

    # Read and convert to a tensor
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]

    hvi_module = RGB_HVI().eval()
    with torch.no_grad():
        hvi = hvi_module.HVIT(img_tensor)  # [1, 3, H, W]

    H, V, I = hvi[:, 0:1], hvi[:, 1:2], hvi[:, 2:3]

    # Print the max values of the H, V, and I channels.
    print(f"[H max] {H.max().item():.4f}")
    print(f"[V max] {V.max().item():.4f}")
    print(f"[I max] {I.max().item():.4f}")

    # Save the three pseudo-color images for H, V, and I.
    for comp, name in zip([H, V, I], ['H', 'V', 'I']):
        vis_img = tensor_to_colormap(comp)
        save_path = os.path.join(save_dir, f'{name}.png')
        cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"[Saved] {save_path}")

    # Generate an HV two-channel pseudo-RGB image.
    hv_rgb = two_channel_to_rgb(H, V)
    hv_path = os.path.join(save_dir, 'HV.png')
    cv2.imwrite(hv_path, cv2.cvtColor(hv_rgb, cv2.COLOR_RGB2BGR))
    print(f"[Saved HV] {hv_path}")


    # ---------------------- Added: YCrCb visualization ----------------------
    # Convert to NumPy format (0–255).
    img_np = np.array(image)
    img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)

    # Split Y, Cr, and Cb
    Y, Cr, Cb = cv2.split(img_ycrcb)
    # Print the maximum value of each YCrCb channel.
    print(f"[Y max] {Y.max()}")
    print(f"[Cr max] {Cr.max()}")
    print(f"[Cb max] {Cb.max()}")

    # Save the Y-channel grayscale image.
    # Apply linear normalization to the range [0, 255].
    Y_norm = cv2.normalize(Y, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Save the enhanced Y-channel image.
    y_enhanced_path = os.path.join(save_dir, 'Y_channel_enhanced.png')
    cv2.imwrite(y_enhanced_path, Y_norm.astype(np.uint8))
    print(f"[Saved Enhanced Y channel] {y_enhanced_path}")

    # Construct a CrCb pseudo-RGB image (Cr → R, Cb → G, B = 0).
    H, W = Cr.shape
    crcb_rgb = np.zeros((H, W, 3), dtype=np.uint8)
    crcb_rgb[:, :, 0] = Cr   # R channel = Cr.
    crcb_rgb[:, :, 1] = Cb   # G channel = Cb.
    # B channel = 0.

    crcb_path = os.path.join(save_dir, 'CrCb.png')
    cv2.imwrite(crcb_path, crcb_rgb)
    print(f"[Saved CrCb] {crcb_path}")


if __name__ == "__main__":
    # Example path: you can replace it with your own image path
    image_path = "/media/HDD/lvyou/BIP-CENet/datasets/LOLv2/Real_captured/Train/Low/00045.png"
    visualize_HVI(image_path)
