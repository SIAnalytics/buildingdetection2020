"""
GT 를 만들기 위해서 Polygon으로 Area 만들기 filling
"""
import os
import sys

import numpy as np
import pandas as pd
# import geopandas as gpd
from skimage.draw import polygon as skpoly
# import torch
import cv2
import json
import matplotlib.pyplot as plt
import csv
from PIL import Image

"""=============================================================================="""
"""                                    parameters                                """
"""=============================================================================="""
## 파일 경로 입력
DIR_geojsons = '/mnt/datasets/RSI_OP_NIA_PUB3/building/label/' 
DIR_csv = '/mnt/datasets/RSI_OP_NIA_PUB3/building/list_building_pub3.csv'
DIR_SAVE_GT_IMG = '/mnt/datasets/RSI_OP_NIA_PUB3/building/label_color_contour_pub3/' 
image_size = 1024
GTformat = True # True: 각 pixel은 class로 기록 (1,2,3,....), False: 각 pixel은 3차원 RGB로 기록

# 저장할 폴더 만들기
if not os.path.exists(DIR_SAVE_GT_IMG):
    os.makedirs(DIR_SAVE_GT_IMG)

# CSV 에서 값 추출
list_geojson = []
f = open(DIR_csv, 'r')
names = csv.reader(f)
for name in names:
    name[0] = name[0] + '.json'  # csv 파일 list 에 확장자가 빠진 이름들의 list 이므로
    list_geojson.append(name[0])
f.close()

data = {'objects':[], 'typeid':[]}
for i in range(list_geojson.__len__()):
    temp = pd.read_json(DIR_geojsons + list_geojson[i], orient='recods')
    buildins_per_img = []
    typeid_of_building = []
    for j in range(temp.shape[0]):
        buildins_per_img.append(temp.values[j][0]['properties']['building_imcoords']) # object_imcoords
        typeid_of_building.append(temp.values[j][0]['properties']['type_id'])

    data['objects'].append(buildins_per_img)
    data['typeid'].append(typeid_of_building)


starting_index = 0

if GTformat == True: # 채널 파라미터 입력
    num_channel = 1  
else:
    num_channel = 3


previous_name = list_geojson[starting_index]
GT_rgb = np.zeros((image_size, image_size, num_channel))

cnt_empty = 0
sum_cnt_m = np.zeros((15,1))

for i in range(starting_index, list_geojson.__len__()): # 전체 건물 갯수에 대해서 반복
    
    cnt_m = np.zeros((15,1))

    # image 이름이 바뀌면 추출한 polygon 정리
    if previous_name != list_geojson[i]: 
        previous_name = list_geojson[i]
        GT_rgb = np.zeros((image_size, image_size, num_channel)) #새로운 container 생성

    temp = data['objects'][i]

    if temp =='EMPTY':
        cnt_empty = cnt_empty + 1 # 잘못된 GT
        raise Exception("This file is not builing GT file.")
    elif (temp == []):
        cnt_empty = cnt_empty + 1 # 건물 없는 영상 체크
        cv2.imwrite(DIR_SAVE_GT_IMG + previous_name.split('.')[0] + '.png', GT_rgb) # save image
    else:
        for j in range(temp.__len__()):
            temp_onepoly = temp[j].split(',')
            polygons = np.zeros((int(temp_onepoly.__len__()/2), 2), np.int32) # polygon 을 float 화 해서 담을 그릇
            for q in range(int(temp_onepoly.__len__()/2)):
                polygons[q, 0] = float(temp_onepoly[q*2]) # 앞쪽 좌표 
                polygons[q, 1] = float(temp_onepoly[q*2+1])
            
            polygons = np.array(polygons)
            temp_id = int(data['typeid'][i][j])

            ''' count '''     
            for q in range(15):
                if temp_id == q+1:
                    cnt_m[q] = cnt_m[q] + 1
                    sum_cnt_m[q] = sum_cnt_m[q] + 1

            ''' Value '''
            if polygons.__len__() > 0: # 건물이 존재
                if GTformat ==  True:
                    if temp_id == 1: # 소형시설 
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (1))
                    elif temp_id == 2: # 아파트
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (2))
                    elif temp_id == 3: # 공장
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (3))
                    elif temp_id == 5: # 중형단독시설
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (4))
                    elif temp_id == 6: # 대형시설
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (5))
                    else: # 4번 container 는 제외
                        _=i # 무의미한 값
                        # raise Exception("building ID missing")
                else:
                    if temp_id == 1: # 소형시설 
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 0, 0))
                    elif temp_id == 2: # 아파트
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (0, 255, 0))
                    elif temp_id == 3: # 공장
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (0, 0, 255))
                    elif temp_id == 5: # 중형단독시설
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (255, 255, 0))
                    elif temp_id == 6: # 대형시설
                        GT_rgb = cv2.fillPoly(GT_rgb, [polygons], (0, 255, 255))
                    else: # 4번 container 는 제외
                        _=i # 무의미한 값
                        # raise Exception("building ID missing")

        name = '_(1)' + str(cnt_m[0]) + '_(2)' + str(cnt_m[1]) + '_(3)' + str(cnt_m[2]) + '_(4)' + str(cnt_m[3]) + '_(5)' + str(cnt_m[4]) + '_(6)' + str(cnt_m[5]) + '_(7)' + str(cnt_m[6]) + '_(8)' + str(cnt_m[7]) + '_(9)' + str(cnt_m[8]) + '_(10)' + str(cnt_m[9]) + '_(11)' + str(cnt_m[10]) + '_(12)' + str(cnt_m[11]) + '_(13)' + str(cnt_m[12]) + '_(14)' + str(cnt_m[13]) + '_(15)' + str(cnt_m[14])

        # cv2.imwrite(DIR_SAVE_GT_IMG + previous_name.split('.')[0] + name + '.png', GT_rgb) # save image
        cv2.imwrite(DIR_SAVE_GT_IMG + previous_name.split('.')[0] + '.png', GT_rgb) # save image

        # ''' 원본도 같이 저장 '''
        # rgb_img = Image.open('/mnt/datasets/RSI_OP_NIA_PUB3/building/asset/' + previous_name.split('.')[0] + '.png').convert('RGB') 
        # rgb_img = np.asarray(rgb_img, np.float32)
        # cv2.imwrite(DIR_SAVE_GT_IMG + previous_name.split('.')[0] + '.jpg', rgb_img) # save image
    
    print('processing: ', i, '/', list_geojson.__len__(), ', num_EMPTY: ', cnt_empty)
    print('processing: ', i, '/', list_geojson.__len__(), ', 1: ', cnt_m[0], ', 2: ', cnt_m[1], ', 3: ', cnt_m[2], ', 4: ', cnt_m[3], ', 5: ', cnt_m[4],
    ', 6: ', cnt_m[5], ', 7: ', cnt_m[6], ', 8: ', cnt_m[7], ', 9: ', cnt_m[8], ', 10: ', cnt_m[9], ', 11: ', cnt_m[10], ', 12: ', cnt_m[11],
    ', 13: ', cnt_m[12], ', 14: ', cnt_m[13], ', 15: ', cnt_m[14]
    )

print('total: ', '1: ', sum_cnt_m[0], ', 2: ', sum_cnt_m[1], ', 3: ', sum_cnt_m[2], ', 4: ', sum_cnt_m[3], ', 5: ', sum_cnt_m[4],
    ', 6: ', sum_cnt_m[5], ', 7: ', sum_cnt_m[6], ', 8: ', sum_cnt_m[7], ', 9: ', sum_cnt_m[8], ', 10: ', sum_cnt_m[9], ', 11: ', sum_cnt_m[10], ', 12: ', sum_cnt_m[11],
    ', 13: ', sum_cnt_m[12], ', 14: ', sum_cnt_m[13], ', 15: ', sum_cnt_m[14]
    )
            


        

    



  
