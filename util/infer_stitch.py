import torch
import numpy as np
from PIL import Image
import torchvision
import matplotlib.pyplot as plt




class InferDiv:
    def __init__(self, size_sample=[225, 255], original_size=[900, 900], x_window_step=5, y_window_step=5):
        super(InferDiv, self).__init__()
        self.size_sample = size_sample
        self.original_size = original_size

        self.x_window_step = x_window_step
        self.y_window_step = y_window_step
        self.x_window_num = int(np.ceil(original_size[0] / x_window_step)) # width
        self.y_window_num = int(np.ceil(original_size[1] / y_window_step)) # height

        self.x_abs_pos = 0  # position x corresponding to original image size
        self.y_abs_pos = 0
        

    def forward(self, img, IMG_MEAN):
        """
        img: original-sized image
        """
        self.info = []
        img = (img - IMG_MEAN)/128
        img = torchvision.transforms.functional.to_tensor(img) # convert to tensor

        cnt = 0
        for i_y in range(self.y_window_num):
            for j_x in range(self.x_window_num):
                x_start = int(j_x * self.x_window_step)
                y_start = int(i_y * self.y_window_step)
                x_range = x_start + self.size_sample[0] # j
                y_range = y_start + self.size_sample[1] # i
  
                # 가장자리 부분
                if (x_range > self.original_size[0]):
                    x_start = self.original_size[0]-self.size_sample[0]
                    x_range = self.original_size[0]
                if (y_range > self.original_size[1]):
                    y_start = self.original_size[1]-self.size_sample[1]
                    y_range = self.original_size[1]

                divided_img = img[:, y_start:y_range, x_start:x_range]
                # divided_img = img.crop((x_start, y_start, x_range, y_range))

                # divided_img = (divided_img*2) -1
                # divided_img = divided_img.transpose(0, 2)
                # divided_img = divided_img.transpose(0, 1)
                # plt.imsave(str(i_y) + '_' + str(j_x) + '.png', divided_img.numpy())

                self.info.append({
                "start_pos_x": x_start,
                "start_pos_y": y_start,
                "div_idx": cnt,
                "divided_img": divided_img
                })

                cnt = cnt +1
        return self.info, cnt



""" sample 결합 """
class InferStitch:
    def __init__(self, size_sample=[224, 224], size_output=[900, 900]):
        super(InferStitch, self).__init__()
        self.size_sample = size_sample
        self.size_output = size_output

    def forward(self, div_img, img_set, num_classes=2):
        w, h = self.size_sample
        W, H = self.size_output
        
        div_img[:, :, :h,  0] = div_img[:, :, :h,  1]
        div_img[:, :, :h, -1] = div_img[:, :, :h, -2]
        div_img[:, :,  0, :w] = div_img[:, :,  1, :w]
        div_img[:, :, -1, :w] = div_img[:, :, -2, :w]

        output = np.zeros((num_classes, H, W))
        divmat = np.zeros((num_classes, h, w))

        for i in range(div_img.shape[0]):
            x, y = img_set[i]['start_pos_x'], img_set[i]['start_pos_y']

            divmat[:] = (output[:, y:y+h, x:x+w] != 0) + 1.
            output[:, y:y+h, x:x+w] += div_img[i, :, :, :]
            output[:, y:y+h, x:x+w] /= divmat

            # print('stitching.... : ', i + 1, '/', div_img.shape[0])

        return output


# class InferStitch:
#     def __init__(self, size_sample=[224, 224], size_output=[900, 900]):
#         super(InferStitch, self).__init__()
#         self.size_sample = size_sample
#         self.size_output = size_output
#         self.x_step = np.ceil(size_output[0] / size_sample[0]) # width
#         self.y_step = np.ceil(size_output[1] / size_sample[1]) # height


#     def forward(self, divided_img, imgset, num_classes=2):
#         """
#         divided_img: sample results
#         """
#         width=self.size_sample[0]
#         height=self.size_sample[1]
#         x_step=self.x_step
#         y_step=self.y_step
        
#         # temp matrix 에 padding
#         output = np.zeros((self.size_output[1], self.size_output[0], num_classes))
#         output[:] = np.nan
#         for i in range(divided_img.shape[0]):
#             temp_map = np.zeros((self.size_output[1], self.size_output[0], num_classes))
#             temp_map[:] = np.nan

#             # image의 가장자리 1 pixel은 convolution 시 나타난 경계로 인한 feature 오류 이므로 바로 옆에 값으로 덮어서 없애준다.
#             divided_img[i, :, 0:height, 0] = divided_img[i, :, 0:height, 1] # 좌
#             divided_img[i, :, 0:height, width-1] = divided_img[i, :, 0:height, width-2] # 우
#             divided_img[i, :, 0, 0:width] = divided_img[i, :, 1, 0:width] # 위
#             divided_img[i, :, height-1, 0:width] = divided_img[i, :, height-2, 0:width] # 아래

#             # divided_img[i,:,:,:] = divided_img[i,:,:,:] + 1000 # 양수로 shift
#             # divided_img[i,0,:,:] = divided_img[i,0,:,:] - divided_img[i,1,:,:] # 테두리의 activation 값을 뺀다
#             # divided_img[i,:,:,:] = divided_img[i,:,:,:] - 1000 # 원래 대로 shift

#             for num in range(num_classes):
#                 temp_map[imgset[i]['start_pos_y']:imgset[i]['start_pos_y']+height, 
#                         imgset[i]['start_pos_x']:imgset[i]['start_pos_x']+width,
#                         num] = divided_img[i, num, :, :]

#             # average stitching
#             output = output.reshape(-1,1)
#             temp_map = temp_map.reshape(-1,1)
#             output = np.nanmean((output, temp_map), axis=0) # nan 이 아닌 경우만 mean을 수행
#             output = output.reshape(self.size_output[1], self.size_output[0], num_classes)

#             print('stitching.... : ', i, '/', divided_img.shape[0])

#         return output
