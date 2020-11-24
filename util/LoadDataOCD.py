
"""=============================================================================="""
"""                                                                              """
"""                         Dataset Load pyplot                                  """
"""                                                                              """
"""=============================================================================="""

import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import json


class LoadsegDBcrop_nia_paper(data.Dataset):
    def __init__(self, root, img_folder_name, filename_list_imgs, label_folder_name, img_size, crop_size=[225, 225], mean=(128, 128, 128), scale=True):
        self.root = root
        self.scale = scale
        self.mean = mean
        self.crop_size = crop_size
        
        self.files = []
        switching = 0

        for i in range(filename_list_imgs.__len__()):
            img_file_post = osp.join(root, "/"+ img_folder_name +"/%s" % filename_list_imgs[i])
            label_file = osp.join(label_folder_name + "/%s" % filename_list_imgs[i])

            if img_size[0] == crop_size[0]:
                left = 0
                top = 0
            else:
                left = np.random.randint(img_size[0] - crop_size[0])
                top = np.random.randint(img_size[1] - crop_size[1])
                
            W = left + crop_size[0]
            H = top + crop_size[1]

            switching = np.random.rand(1)

            self.files.append({
                "img_rgb": img_file_post,
                "label": label_file,
                "name": filename_list_imgs[i].split('.')[0],
                "left": left,
                "top": top,
                "W": W,
                "H": H,
                "switching": switching
            })

    def __len__(self):
        return len(self.files)


    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_rgb = Image.open(self.root + datafiles["img_rgb"]).convert('RGB') 

        if datafiles["switching"] >= 0.5:
            img_rgb = img_rgb.transpose(Image.ROTATE_180)

        # Crop
        crop_img = img_rgb.crop((datafiles["left"], datafiles["top"], datafiles["W"], datafiles["H"]))
        crop_img = np.asarray(crop_img, np.float32)
        # size = img_rgb.shape
        crop_img = (crop_img - self.mean)/128

        return crop_img.copy(), datafiles, index


class LoadDBcrop(data.Dataset):
    def __init__(self, root, sar_folder_name, fake_folder_name, label_list, label_foler_name, img_size, crop_size=[225, 225], mean=(128, 128, 128), scale=True):
        self.root = root
        self.scale = scale
        self.mean = mean
        self.crop_size = crop_size
        
        self.files = []
        switching = 0

        for i in range(label_list.__len__()):
            sar_file_name = osp.join(root, sar_folder_name +"/%s" % label_list[i])
            fake_file_name = osp.join(root, fake_folder_name +"/%s" % label_list[i])
            label_file = osp.join(root, label_foler_name + "/%s" % label_list[i])

            left = np.random.randint(img_size[0] - crop_size[0])
            top = np.random.randint(img_size[1] - crop_size[1])
            W = left + crop_size[0]
            H = top + crop_size[1]

            switching = np.random.rand(1)

            self.files.append({
                "img_sar": sar_file_name,
                "img_fake": fake_file_name,
                "label": label_file,
                "name": label_list[i].split('.')[0],
                "left": left,
                "top": top,
                "W": W,
                "H": H,
                "switching": switching
            })

    def __len__(self):
        return len(self.files)

    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_sar = Image.open(datafiles["img_sar"]).convert('RGB') 
        img_fake = Image.open(datafiles["img_fake"]).convert('RGB')

        # normal -> 180 
        if datafiles["switching"] >= 0.5:
            img_sar = img_sar.transpose(Image.ROTATE_180)
            img_fake = img_fake.transpose(Image.ROTATE_180)

        # Crop
        crop_img_sar = img_sar.crop((datafiles["left"], datafiles["top"], datafiles["W"], datafiles["H"]))
        crop_img_fake = img_fake.crop((datafiles["left"], datafiles["top"], datafiles["W"], datafiles["H"]))
        crop_img_sar = np.asarray(crop_img_sar, np.float32)
        crop_img_fake = np.asarray(crop_img_fake, np.float32)
        crop_img_sar = (crop_img_sar - self.mean)/128
        crop_img_fake = (crop_img_fake - self.mean)/128

        return crop_img_sar, crop_img_fake, datafiles, index

class LoadsegDBSarFake(data.Dataset):
    def __init__(self, root, sar_folder_name, fake_folder_name, label_list, label_foler_name, img_size, crop_size=[225, 225], mean=(128, 128, 128), scale=True):
        self.root = root
        self.scale = scale
        self.mean = mean
        self.crop_size = crop_size
        
        self.files = []
        switching = 0

        for i in range(label_list.__len__()):
            sar_file_name = osp.join(root, sar_folder_name +"/%s" % label_list[i])
            fake_file_name = osp.join(root, fake_folder_name +"/%s" % label_list[i])
            label_file = osp.join(root, label_foler_name + "/%s" % label_list[i])

            left = np.random.randint(img_size[0] - crop_size[0])
            top = np.random.randint(img_size[1] - crop_size[1])
            W = left + crop_size[0]
            H = top + crop_size[1]

            switching = np.random.rand(1)

            self.files.append({
                "img_sar": sar_file_name,
                "img_fake": fake_file_name,
                "label": label_file,
                "name": label_list[i].split('.')[0],
                "left": left,
                "top": top,
                "W": W,
                "H": H,
                "switching": switching
            })

    def __len__(self):
        return len(self.files)


    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_sar = Image.open(datafiles["img_sar"]).convert('RGB') 
        img_fake = Image.open(datafiles["img_fake"]).convert('RGB')

        # normal -> 180 
        if datafiles["switching"] >= 0.5:
            img_sar = img_sar.transpose(Image.ROTATE_180)
            img_fake = img_fake.transpose(Image.ROTATE_180)

        # Crop
        crop_img_sar = img_sar.crop((datafiles["left"], datafiles["top"], datafiles["W"], datafiles["H"]))
        crop_img_fake = img_fake.crop((datafiles["left"], datafiles["top"], datafiles["W"], datafiles["H"]))
        crop_img_sar = np.asarray(crop_img_sar, np.float32)
        crop_img_fake = np.asarray(crop_img_fake, np.float32)
        crop_img_sar = (crop_img_sar - self.mean)/128
        crop_img_fake = (crop_img_fake - self.mean)/128

        crop_img = np.concatenate((crop_img_sar, crop_img_fake), 2)

        return crop_img.copy(), datafiles, index


def LoadforAug(root, post_folder_name, label_list, label_foler_name):
    root = root       
    files = []

    for i in range(label_list.__len__()):
        img_file_post = osp.join(root, "/"+ post_folder_name +"/%s" % label_list[i])
        label_file = osp.join(root, "/" + label_foler_name + "/%s" % label_list[i])

        files.append({
            "img_rgb": img_file_post,
            "label": label_file,
            "name": label_list[i].split('.')[0]
        })

    return files



class LoadsegDBcrop_nia(data.Dataset):
    def __init__(self, root, post_folder_name, label_list, label_foler_name, name_list_mask, img_size, crop_size=[225, 225], mean=(128, 128, 128), scale=True):
        self.root = root
        self.scale = scale
        self.mean = mean
        self.crop_size = crop_size
        
        self.files = []
        switching = 0

        for i in range(label_list.__len__()):
            img_file_post = osp.join(root, "/"+ post_folder_name +"/%s" % label_list[i])
            label_file = osp.join(label_foler_name + "/%s" % label_list[i])

            left = np.random.randint(img_size[0] - crop_size[0])
            top = np.random.randint(img_size[1] - crop_size[1])
            W = left + crop_size[0]
            H = top + crop_size[1]

            switching = np.random.rand(1)

            self.files.append({
                "img_rgb": img_file_post,
                "label": label_file,
                "name": name_list_mask[i].split('.')[0],
                "left": left,
                "top": top,
                "W": W,
                "H": H,
                "switching": switching
            })
        #     print("================================================================")
        #     print(img_file)
        #     print(label_file)
        #     print("================================================================")
        # print("break")

    def __len__(self):
        return len(self.files)


    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_rgb = Image.open(self.root + datafiles["img_rgb"]).convert('RGB') 

        if datafiles["switching"] >= 0.5:
            img_rgb = img_rgb.transpose(Image.ROTATE_180)

        # Crop
        crop_img = img_rgb.crop((datafiles["left"], datafiles["top"], datafiles["W"], datafiles["H"]))
        crop_img = np.asarray(crop_img, np.float32)
        # size = img_rgb.shape
        crop_img = (crop_img - self.mean)/128

        return crop_img.copy(), datafiles, index




class LoadsegDBcrop(data.Dataset):
    def __init__(self, root, post_folder_name, label_list, label_foler_name, img_size, crop_size=[225, 225], mean=(128, 128, 128), scale=True):
        self.root = root
        self.scale = scale
        self.mean = mean
        self.crop_size = crop_size
        
        self.files = []
        switching = 0

        for i in range(label_list.__len__()):
            img_file_post = osp.join(root, "/"+ post_folder_name +"/%s" % label_list[i])
            label_file = osp.join(root, "/" + label_foler_name + "/%s" % label_list[i])

            left = np.random.randint(img_size[0] - crop_size[0])
            top = np.random.randint(img_size[1] - crop_size[1])
            W = left + crop_size[0]
            H = top + crop_size[1]

            switching = np.random.rand(1)

            self.files.append({
                "img_rgb": img_file_post,
                "label": label_file,
                "name": label_list[i].split('.')[0],
                "left": left,
                "top": top,
                "W": W,
                "H": H,
                "switching": switching
            })
        #     print("================================================================")
        #     print(img_file)
        #     print(label_file)
        #     print("================================================================")
        # print("break")

    def __len__(self):
        return len(self.files)


    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_rgb = Image.open(self.root + datafiles["img_rgb"]).convert('RGB') 

        if datafiles["switching"] >= 0.5:
            img_rgb = img_rgb.transpose(Image.ROTATE_180)

        # Crop
        crop_img = img_rgb.crop((datafiles["left"], datafiles["top"], datafiles["W"], datafiles["H"]))
        crop_img = np.asarray(crop_img, np.float32)
        # size = img_rgb.shape
        crop_img = (crop_img - self.mean)/128

        return crop_img.copy(), datafiles, index


def LoadforAug(root, post_folder_name, label_list, label_foler_name):
    root = root       
    files = []

    for i in range(label_list.__len__()):
        img_file_post = osp.join(root, "/"+ post_folder_name +"/%s" % label_list[i])
        label_file = osp.join(root, "/" + label_foler_name + "/%s" % label_list[i])

        files.append({
            "img_rgb": img_file_post,
            "label": label_file,
            "name": label_list[i].split('.')[0]
        })

    return files


class LoadsegDB(data.Dataset):
    def __init__(self, root, post_folder_name, label_list, label_foler_name, re_size=(1024, 1024), mean=(128, 128, 128), scale=True):
        self.root = root
        self.re_size = re_size
        self.scale = scale
        self.mean = mean
        
        self.files = []

        for i in range(label_list.__len__()):
            img_file_post = osp.join(root, "/"+ post_folder_name +"/%s" % label_list[i])
            label_file = osp.join(root, "/" + label_foler_name + "/%s" % label_list[i])

            self.files.append({
                "img_rgb": img_file_post,
                "label": label_file,
                "name": label_list[i].split('.')[0]
            })
        #     print("================================================================")
        #     print(img_file)
        #     print(label_file)
        #     print("================================================================")
        # print("break")

    def __len__(self):
        return len(self.files)


    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_rgb = Image.open(self.root + datafiles["img_rgb"]).convert('RGB') 
 
        # label_json_info = json.load(self.root + datafiles["label"])
        # data_name = datafiles["name"]

        # resize
        img_rgb = img_rgb.resize(self.re_size, Image.BICUBIC)
        img_rgb = np.asarray(img_rgb, np.float32)

        # size = img_rgb.shape
        img_rgb = (img_rgb - self.mean)/128

        return img_rgb.copy(), datafiles, index


class LoadxViewDB(data.Dataset):
    def __init__(self, root, image_list_png, label_list_json, max_iters=None, re_size=(1024, 1024), mean=(128, 128, 128), scale=True, mirror=True):
        self.root_prepost = root + '/post'
        self.re_size = re_size
        self.scale = scale
        self.mean = mean
        self.is_mirror = mirror
        
        self.files = []

        for i in range(image_list_png.__len__()):
            img_file = osp.join(self.root_prepost, "/images/%s" % image_list_png[i])
            label_file = osp.join(self.root_prepost, "/labels/%s" % label_list_json[i])

            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_list_png[i].split('_p')[0]
            })
        #     print("================================================================")
        #     print(img_file)
        #     print(label_file)
        #     print("================================================================")
        # print("break")
    def __len__(self):
        return len(self.files)


    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img = Image.open(self.root_prepost + datafiles["img"]).convert('RGB') 
        # label_json_info = json.load(self.root + datafiles["label"])
        # data_name = datafiles["name"]

        # resize
        img = img.resize(self.re_size, Image.BICUBIC)
        img = np.asarray(img, np.float32)

        size = img.shape
        img -= self.mean

        return img.copy(), datafiles, index
        # return img.copy(), label_json_info, np.array(size), data_name


class LoadTestImg(data.Dataset):
    def __init__(self, root, image_list_png, re_size=(1024, 1024), mean=(128, 128, 128), scale=True):
        self.root_prepost = root + '/post'
        self.re_size = re_size
        self.scale = scale
        self.mean = mean
        self.files = []

        for i in range(image_list_png.__len__()):
            img_file = osp.join(self.root_prepost, "/%s" % image_list_png[i])

            self.files.append({
                "img": img_file,
                "name": image_list_png[i].split('_p')[0]
            })


    def __len__(self):
        return len(self.files)


    # data.DataLoader 에서 실행
    def __getitem__(self, index):
        datafiles = self.files[index]
        img = Image.open(self.root_prepost + datafiles["img"]).convert('RGB') 
  
        # resize
        img = img.resize(self.re_size, Image.BICUBIC)
        img = np.asarray(img, np.float32)

        size = img.shape
        img -= self.mean

        return img.copy(), datafiles, index
