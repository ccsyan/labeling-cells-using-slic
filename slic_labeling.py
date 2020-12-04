# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np
import cv2
import pickle
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from scipy.ndimage.morphology import binary_dilation as dila

def bwRGB(bw,im):
    A = np.sum(bw)
    B = np.sum(im[bw,0])/A
    G = np.sum(im[bw,1])/A
    R = np.sum(im[bw,2])/A
    return [R,G,B]

def col_dis(color1,color2):
    sum = 0
    for i in range(3):
        ds = (float(color1[i]) - float(color2[i]))**2
        sum =sum+ds
    delta_e = np.sqrt(sum)
    return delta_e

path0 = './'
#%% colloct image file names     
files = []
for folderName, subfolders, filenames in os.walk(path0):
    for filename in filenames:
        if 'tif' in filename and folderName == path0:
            files.append(folderName+'/'+filename)

print('total files:', len(files))
#%% slic each image

# these are two parameters as color space distance, determined by experiences
merge = 12 
dark =  40

for i in range(len(files)):

    im0 = cv2.imread(files[i])
    
    seg0 = slic(im0, n_segments = 500,
                    multichannel=True,
                    convert2lab=True,
                    enforce_connectivity=True,
                    slic_zero=False, compactness=30,
                    max_iter=100,              
                    sigma = [0,1.7,1.7],
                    spacing=[0,1,1],#z,y,x
                    min_size_factor=0.4,
                    max_size_factor=3,
                    start_label=0)
    # parameters can refer to https://www.kite.com/python/docs/skimage.segmentation.slic

    f = open(path0+'im'+str(i)+'.seg0.pkl','wb')
    pickle.dump(seg0,f)
    f.close()
    
    im1 = np.uint8(mark_boundaries(im0, seg0)*255)
    cv2.imwrite(path0+'im'+str(i)+'.seg0.png',im1)

    #%% first time merging neighbors
    labels = np.unique(seg0)
    seg1 = seg0.copy()
    lindex=501 # new labels on seg1 starts from 501
    for label in labels:
        if label > 0 and label < 900:
            bw = seg1 == label 
            A = np.sum(bw)
            if A > 0:
                color1 = bwRGB(bw,im0)
                color_dist = col_dis(color1,[0,0,0])
                if color_dist < dark:
                    seg1[seg1==label] = 0 # dark region on seg1 is labeled as 0
                else:
                    seg1[seg1==label] = lindex 
                    # looking for neighbors
                    bwd = dila(bw)
                    nlabels=np.unique(seg1[bwd]) # neibor's labels
                    for nl in nlabels:
                        if nl > label and nl < 500:
                            bw2 = seg1 ==nl
                            color2 = bwRGB(bw2,im0)
                            if col_dis(color1,color2) < merge: 
                                seg1[seg1==nl]=lindex 
                lindex +=1 
    
    
    f = open(path0+'im'+str(i)+'.seg1.pkl','wb')
    pickle.dump(seg1,f)
    f.close()
        
    im1 = np.uint8(mark_boundaries(im0, seg1)*255)
    cv2.imwrite(path0+'im'+str(i)+'.seg1.png',im1)
