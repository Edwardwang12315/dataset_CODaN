import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import scipy.stats as stats
import random
import gc  # 垃圾回收模块

# 设置设备
device = torch.device("cpu")

# 设置数据集
CODaN = True
WiderFace = False

def apply_ccm(image, ccm): # device不影响
    shape = image.shape
    image = image.view(-1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    return image.view(shape)

def random_noise_levels(): # device不影响
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = np.random.uniform(log_min_shot_noise, log_max_shot_noise)
    shot_noise = np.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    log_read_noise = line(log_shot_noise) + np.random.normal(scale=0.26)
    read_noise = np.exp(log_read_noise)
    return shot_noise, read_noise

def Low_Illumination_Degrading(img, safe_invert=False): # device不影响

    img = img.to(device)
    
    config = dict(darkness_range=(0.01, 0.1),
                 gamma_range=(2.0, 3.5),
                 rgb_range=(0.8, 0.1),
                 red_range=(1.9, 2.4),
                 blue_range=(1.5, 1.9),
                 quantisation=[12, 14, 16])
    
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                 [-0.5625, 1.6328, -0.0469],
                 [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                 [-0.613, 1.3513, 0.2906],
                 [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                 [-0.2887, 1.0725, 0.2496],
                 [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                 [-0.4782, 1.3016, 0.1933],
                 [-0.097, 0.1581, 0.5181]]]
    rgb2xyz = [[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]]

    img1 = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    img1 = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * img1) / 3.0)
    
    epsilon = torch.tensor([1e-8], dtype=torch.float, device=device)
    gamma = random.uniform(config['gamma_range'][0], config['gamma_range'][1])
    img2 = torch.max(img1, epsilon) ** gamma
    
    xyz2cam = random.choice(xyz2cams)
    rgb2cam = np.matmul(xyz2cam, rgb2xyz)
    rgb2cam = torch.from_numpy(rgb2cam / np.sum(rgb2cam, axis=-1)).float().to(device)
    img3 = apply_ccm(img2, rgb2cam)
    
    rgb_gain = random.normalvariate(config['rgb_range'][0], config['rgb_range'][1])
    red_gain = random.uniform(config['red_range'][0], config['red_range'][1])
    blue_gain = random.uniform(config['blue_range'][0], config['blue_range'][1])

    gains1 = np.stack([1.0 / red_gain, 1.0, 1.0 / blue_gain]) * rgb_gain
    gains1 = gains1[np.newaxis, np.newaxis, :]
    gains1 = torch.tensor(gains1, dtype=torch.float, device=device)

    if safe_invert:
        img3_gray = torch.mean(img3, dim=-1, keepdim=True)
        inflection = 0.9
        zero = torch.zeros_like(img3_gray, device=device)
        mask = (torch.max(img3_gray - inflection, zero) / (1.0 - inflection)) ** 2.0
        safe_gains = torch.max(mask + (1.0 - mask) * gains1, gains1)
        img4 = torch.clamp(img3 * safe_gains, min=0.0, max=1.0)
    else:
        img4 = img3 * gains1

    # 低光照处理
    lower, upper = config['darkness_range'][0], config['darkness_range'][1]
    mu, sigma = 0.1, 0.08
    darkness = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    darkness = darkness.rvs()
    img5 = img4 * darkness
    
    shot_noise, read_noise = random_noise_levels()
    var = img5 * shot_noise + read_noise
    var = torch.max(var, epsilon)
    noise = torch.normal(mean=0, std=torch.sqrt(var))
    img6 = img5 + noise

    # ISP处理
    bits = random.choice(config['quantisation'])
    quan_noise = torch.tensor(img6.size(), dtype=torch.float, device=device).uniform_(-1/(255*bits), 1/(255*bits))
    img7 = img6 + quan_noise
    
    gains2 = np.stack([red_gain, 1.0, blue_gain])
    gains2 = gains2[np.newaxis, np.newaxis, :]
    gains2 = torch.tensor(gains2, dtype=torch.float, device=device)
    img8 = img7 * gains2
    
    cam2rgb = torch.inverse(rgb2cam)
    img9 = apply_ccm(img8, cam2rgb)
    img10 = torch.max(img9, epsilon) ** (1 / gamma)
    
    img_low = img10.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    para_gt = torch.tensor([darkness, 1.0 / gamma, 1.0 / red_gain, 1.0 / blue_gain], dtype=torch.float, device=device)
    
    return img_low, para_gt

if WiderFace:
    def process_images(list_file, data_root, dst_root):
        """ 
        list_file = "../WiderFace/wider_face_train_self.txt" 记录图片目录(可以不完全正确,只要子文件夹对应正确即可)和gt值的文件
        data_root = "../WiderFace" 记录原始图片的总文件夹位置,使用时用root+子文件夹定位
        dst_root = "./WIDER_train/images" 记录当前任务的目标子文件夹，使用时需要把本函数放在目标文件夹目录下
        """
        with open(list_file, 'r') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc=f"Processing {os.path.basename(list_file)}"): # 使用GPU可以设置batch等
            file = line.strip().split()[0]
            if 'WiderFace/' in file:
                file = file.split('WiderFace/')[1]
            
            filepath = os.path.join(data_root, file)
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
            
            # 创建目标目录
            foldname = os.path.dirname(file).split('/')[-1]
            foldpath = os.path.join(dst_root, foldname)
            os.makedirs(foldpath, exist_ok=True)
            
            # 读取并处理图像
            img = cv2.imread(filepath)
            if img is None:
                print(f"Failed to read image: {filepath}")
                continue
                
            # 转换为RGB并归一化
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # 应用低光照处理
            with torch.no_grad():
                dark_img, _ = Low_Illumination_Degrading(img_tensor)
            
            # 转换回OpenCV格式
            dark_img = dark_img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
            dark_img = np.clip(dark_img * 255, 0, 255).astype(np.uint8)
            dark_img_bgr = cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR)
            
            # 保存处理后的图像
            filename = os.path.basename(file)
            save_path = os.path.join(foldpath, filename)
            cv2.imwrite(save_path, dark_img_bgr)
            # exit()
            
if CODaN:
    def process_images(list_file, data_root, dst_root):
        """ 
        list_file = "../CODaN/data/train" 记录原始图片类别文件夹上层路径
        data_root = 
        dst_root = "./data/train" 记录目标所在子文件夹，使用时需要把本函数放在目标文件夹目录下
        """

        categories = [d for d in os.listdir(list_file) if os.path.isdir(os.path.join(list_file, d))]
        for category in tqdm(categories , desc= f'正在处理文件{list_file}'):
            
            path_read = os.path.join(list_file,category)
            path_save = os.path.join(dst_root,category)
            os.makedirs(path_save,exist_ok=True)

            file_names = [ f for f in os.listdir(path_read) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for file_name in tqdm(file_names,desc=f'正在处理类别{category}'):
                file_path = os.path.join(list_file,category,file_name)
                save_path = os.path.join(dst_root,category,file_name)
                img = cv2.imread(file_path)

                # 转换为RGB并归一化
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                
                # 应用低光照处理
                with torch.no_grad():
                    dark_img, _ = Low_Illumination_Degrading(img_tensor)
                
                # 转换回OpenCV格式
                dark_img = dark_img.permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                dark_img = np.clip(dark_img * 255, 0, 255).astype(np.uint8)
                dark_img_bgr = cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR)
                
                # 保存处理后的图像
                cv2.imwrite(save_path, dark_img_bgr)

# 路径配置
list_file1 = "./data/train"
list_file2 = "./data/val"
data_root = "../WiderFace"
dst_root1 = "./data_dark/train/"
dst_root2 = "./data_dark/val/"

# 处理训练集和验证集
process_images(list_file1, data_root, dst_root1)
process_images(list_file2, data_root, dst_root2)