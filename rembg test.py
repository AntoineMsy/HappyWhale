# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:51:41 2022

@author: antoi
"""

from rembg import remove
import cv2

input_path = "train_images/000a8f2d5c316a.jpg"
output_path = 'output.png'
input_path_recrop = 'input_recrop.png'

input = cv2.imread(input_path)
print(input.shape)
output = remove(input)
max_abs = 0
min_abs = output.shape[0]
max_ord = 0
min_ord = output.shape[1]
print(min_abs,max_abs,min_ord,max_ord)

print(output.shape)
for i in range(output.shape[0]):
    for j in range(output.shape[1]):
        isNotNull = False
        s = 0
        for k in range(output.shape[2]):
            s+= output[i,j,k]
        if s != 0:
            isNotNull = True
        if isNotNull :
            if i< min_abs :
                min_abs = i
            if i> max_abs:
                max_abs = i
            if j< min_ord :
                min_ord = j
            if j>max_ord:
                max_ord = j
print(min_abs,max_abs,min_ord,max_ord)
            
output = output[min_abs:max_abs,min_ord:max_ord,:]
input = input[min_abs:max_abs,min_ord:max_ord,:]
cv2.imwrite(output_path, output)
cv2.imwrite(input_path_recrop, input)