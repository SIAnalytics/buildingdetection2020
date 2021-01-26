import torch
import torch.nn as nn
from torch.utils import data

from torch.autograd import Variable
import torch.optim as optim

import torch.backends.cudnn as cudnn

import argparse
import numpy as np
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import json
from PIL import Image
import csv
import time

from scipy.special import softmax

from torchvision import models
from util.iouEval import iouEval, getColorEntry
from util.infer_stitch import InferDiv, InferStitch


""""=============================================================================="""
"""                             GPU device 설정                                   """
"""==============================================================================="""
gpu_device = '3' # GPU 번호
model_mode = 'DeepRes101' # DeepRes101 / DeepRes50 / DeepFCN101
"""=============================================================================="""
"""                                                                              """
"""                             주요 파라미터 셋팅                                 """
"""                                                                              """
"""=============================================================================="""

"""------------------------ LOAD ---------------------------"""
DATA_DIR = '/mnt/datasets/RSI_OP_NIA_PUB4/building'
img_folder_name = 'asset' 
label_folder_name = 'label_gray_contour_pub4' #'trainLabel' #'GT_rgbseg'
GT_FORMAT = '.png'
INPUT_SIZE = 1024 # 학습으로 들어가는 영상 크기
ORIGINAL_SIZE = 1024 # 데이터셋 영상 크기 
x_window_step = 200
y_window_step = 200
BACKBONE_DIR = '/mnt/workspace/hyunguk/nia/snapshots_nia_building_resnet101_contour_1386_modi/'
BACKBONE_NAME = 'sn6_resunet50_148.pth'
csv_data = 'list_building_test_pub4_modi_list1.csv'
"""------------------------ SAVE ---------------------------"""
RESULTS_DIR = '/mnt/workspace/hyunguk/nia/RESULTS_nia_deepRes50_1386_ep149/' # 저장 경로
"""------------------------ function parameters ---------------------------"""
NUM_CLASSES = 7  # 0:배경, 1:소형, 2:아파트, 3:공장, 4:중형단독시설, 5:대형시설, 6:contour
trans_ratio = 0.5 # save 투영율 (1일수록 짙은 label)

"""=============================================================================="""
"""                                                                              """
"""                             Parser                                           """
"""                                                                              """
"""=============================================================================="""
def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='BUILDING DETECTION')   
    parser.add_argument("--restore-from", type=str, default=BACKBONE_DIR,
                        help="DIR to load weight file")
    parser.add_argument("--backbone-name", type=str, default=BACKBONE_NAME,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--list-file_name", type=str, default=csv_data,
                        help="Dataset list for test")
    parser.add_argument("--img-folder-name", type=str, default=img_folder_name,
                        help="folder name including image data")
    parser.add_argument("--label-folder-name", type=str, default=label_folder_name,
                        help="folder name including GT data")
    parser.add_argument("--gt-format", type=str, default=GT_FORMAT,
                        help="Input ground truth format")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="The size of input image for training")
    parser.add_argument("--imgdata-size", type=int, default=ORIGINAL_SIZE,
                        help="The size of image in the dataset")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--random-scale", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--model-mode", type=str, default=model_mode,
                        help="Models : DeepRes101 / DeepRes50 / DeepFCN101")
    parser.add_argument("--gpu", type=str, default=gpu_device,
                        help="choose gpu device.")

    return parser.parse_args()

args = get_arguments()

os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
IMG_MEAN = np.array((128, 128, 128), dtype=np.float32) # 학습 속도를 위해 RGB 128을 영점으로 둔다. [-128~127], Load code에서 128로 나눔 [-1~0.999]
INPUT_SIZE_m = [args.input_size, args.input_size] 
original_size = [args.imgdata_size, args.imgdata_size]  # width, height

"""=============================================================================="""
"""                                                                              """
"""                                     MAIN                                     """
"""                                                                              """
"""=============================================================================="""
def main():
    cudnn.enabled = True

    """-------------------------- 개발 MODEL ROAD --------------------------"""
    if args.model_mode == 'DeepRes101':
        model = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=args.num_classes)
    elif args.model_mode == 'DeepRes50':
        model = models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=args.num_classes)
    elif args.model_mode == 'DeepFCN101':
        model = models.segmentation.fcn_resnet101(pretrained=False, num_classes=args.num_classes)
    else:
        raise Exception("Please select a model")

    model.cuda(0)
    # model=nn.DataParallel(model)
    model.load_state_dict(torch.load(args.restore_from + args.backbone_name))
    model.eval()
    
    # 쉽게 true로 두면 비용(memory 등) 이 더 들지만 성능이 향상됨.
    cudnn.benchmark = False  # cudnn.benchmark = true -- uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
                            # -- If this is set to false, uses some in-built heuristics that might not always be fastest.

    """-------------------------- FILE SAVE --------------------------"""
    if not os.path.exists(args.results_dir + 'featuremap/'):
        os.makedirs(args.results_dir + 'featuremap/')
        
    """-------------------------- FILE LOAD ----------------------------------"""
    name_list = []
#     f = open(args.data_dir + args.list_file_name, 'r')
    f = open(args.list_file_name, 'r')
    names = csv.reader(f)
    for name in names:
        name[0] = name[0] + args.gt_format  # csv 파일 list 에 확장자가 빠진 이름들의 list 이므로
        name_list.append(name[0])
    f.close() 

    divided_img = InferDiv(size_sample=INPUT_SIZE_m, original_size=original_size, x_window_step=x_window_step, y_window_step=y_window_step)
    stitching_img = InferStitch(size_sample=INPUT_SIZE_m, size_output=original_size) 
    if args.num_classes > 6:
        fEval = iouEval(nClasses=6, ignoreIndex=-1) # iou 선언
    else:
        fEval = iouEval(nClasses=args.num_classes, ignoreIndex=-1) # iou 선언
    """-------------------------- TEST START ----------------------------------"""
    start = time.time()
    for cntList in range(name_list.__len__()):
        # image open
        img_file = args.data_dir + "/"+ args.img_folder_name +"/%s" % name_list[cntList]
        img_rgb = Image.open(img_file).convert('RGB') 
        # # label open
        label_file = args.data_dir + "/"+ args.label_folder_name +"/%s" % name_list[cntList]
        label_building = Image.open(label_file).convert('RGB') 

        labels = np.zeros((original_size[0], original_size[1], args.num_classes), np.float32)
        label_building = np.asarray(label_building, np.float32)
        for i in range(args.num_classes):
            [idx_i, idx_j] = np.where(label_building[:,:,2] == i)  
            labels[idx_i, idx_j, i] = 1.0 #building ID

        ''' 영상 crop 해서 계산 한 후 다시 붙여 주는 부분이 필요 (image 만 계산) '''
        imgset, num_patches = divided_img.forward(img_rgb, IMG_MEAN) # 영상을 crop size 로 쪼개는 부분

        for cnt_patches in range(num_patches):
            onepatch = imgset[cnt_patches]['divided_img']
            onepatch = onepatch.expand(1, 3, INPUT_SIZE_m[0], INPUT_SIZE_m[1])
            if cnt_patches == 0:
                sets = onepatch
            else:
                sets = torch.cat((sets, onepatch), 0)
        image_input = Variable(sets).cuda(0)

        ''' 모델 연산 '''
        with torch.no_grad():
            pred_model = model(image_input) 
            pred_model = pred_model['out']
         
        pred_model = pred_model.cpu().detach().numpy()

        ''' output 결과들을 병합 '''
        results = stitching_img.forward(pred_model, imgset, num_classes=args.num_classes)
        results = softmax(results, axis=0)
        
        if results.shape[0] > 6: # conture 제거 작업
            con_idx_i, con_idx_j = np.where(results[6,:,:] >= 0.5)
            results[0,con_idx_i, con_idx_j] = 1 # conture 위치에 1을 주어서 배경으로 바꿈
            results = results[0:-1,:,:]
            labels = labels[:,:,0:-1]

        results = np.expand_dims(results, axis=0)
        results = torch.tensor(results)
        results = Variable(results).cuda(0)

        labels = np.asarray(labels)
        labels = np.expand_dims(labels, axis=0)
        labels = torch.tensor(labels)
        labels = labels.transpose(1, 3)
        labels = labels.transpose(2, 3)
        labels = Variable(labels).cuda(0)
        fEval.addBatch(results.data, labels)

        """-------------------------- RESULT SAVE ----------------------------------"""
        results = results.cpu().detach().numpy()
        # results = softmax(results, axis=1)
        output_save = np.zeros((original_size[0], original_size[1], 3))

        if args.num_classes > 6:
            cc_range = 6
        else:
            cc_range = args.num_classes
        for cc in range(cc_range):
            temp_i, temp_j = np.where(np.array(results[0, cc, :, :]) >= 0.5)
            output_save[temp_i, temp_j, 0] = np.float(np.uint8(70*cc)/255)
            output_save[temp_i, temp_j, 1] = np.float(np.uint8(100*cc)/255)
            output_save[temp_i, temp_j, 2] = np.float(np.uint8(150*cc)/255)

        img_rgb = np.asarray(img_rgb, np.float32)
        merged_results = trans_ratio*(img_rgb/255) + (1 - trans_ratio)*output_save

        label_load = '/mnt/datasets/RSI_OP_NIADacon/building/label_seg_color_1386' +"/%s" % name_list[cntList]
        label_save = Image.open(label_load)
        label_save = np.asarray(label_save, np.float32)

        # plt.imsave(args.results_dir + 'featuremap/' +'GT_' + name_list[cntList] , label_save/255)
        # plt.imsave(args.results_dir + 'featuremap/' +'FM_' + name_list[cntList] , merged_results)

        # # plt.imsave(args.results_dir + 'featuremap/' +'FM_' + name_list[cntList] , results[:, :, 0], cmap='jet')

        print('progress: ', cntList, '/', name_list.__len__())


        # if cntList == 10:
        #     break


    # 각 image 에 대해서 계산 끝난 후 평가
    iouVal, iou_classes = fEval.getIoU()
    acc_class, acc_binary = fEval.getAcc()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i])+'{:0.2f}'.format(iou_classes[i]*100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took ", time.time()-start, "seconds")
    print("=======================================")
    #print("TOTAL IOU: ", iou * 100, "%")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "배경")
    print(iou_classes_str[1], "소형")
    print(iou_classes_str[2], "아파트")
    print(iou_classes_str[3], "공장")
    print(iou_classes_str[4], "중형단독")
    print(iou_classes_str[5], "대형")

    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print("MEAN IoU: ", iouStr, "%")
    print("=======================================")
    print("Per-Class ACC:")
    print(acc_class[0], "배경")
    print(acc_class[1], "소형")
    print(acc_class[2], "아파트")
    print(acc_class[3], "공장")
    print(acc_class[4], "중형단독")
    print(acc_class[5], "대형")

    print ("ACC binary: ", acc_binary, "%")
    print("=======================================")

if __name__ == '__main__':
    main()
