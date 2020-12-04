# -*- coding: utf-8 -*-
"""
"""

import os
import numpy as np
import cv2
import pickle
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops

def edgeremove(seg5): # removing segments on the edge
    labels = list(np.unique(seg5))

    up_edge = np.unique(seg5[0,:])
    for label in up_edge:
        try:
            labels.remove(label)
        except:
            continue

    down_edge = np.unique(seg5[-1,:])
    for label in down_edge:
        try:
            labels.remove(label)
        except:
            continue

    left_edge = np.unique(seg5[:,0])
    for label in left_edge:
        try:
            labels.remove(label)
        except:
            continue

    right_edge = np.unique(seg5[:,-1])
    for label in right_edge:
        try:
            labels.remove(label)
        except:
            continue
    return(labels)

def bwim2(bw,im,name,ratio):
    im2 = np.uint8(mark_boundaries(im, bw,color=(0, 0, 1),
                                   outline_color=(0,0,1))*255)
    cv2.namedWindow(name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name,int(im.shape[1]*ratio),int(im.shape[0]*ratio))
    if name == '1':
        cv2.moveWindow(name,0,0)
    else:
        cv2.moveWindow(name,int(im.shape[1]*ratio)+1,0)
    cv2.imshow(name,im2)
    cv2.waitKey(10)

from win32api import GetSystemMetrics
height = GetSystemMetrics(1) * 0.88
width = GetSystemMetrics(0)

#%%
path0 = './'

# pick the files
files = []
for folderName, subfolders, filenames in os.walk(path0):
    for filename in filenames:
        if 'tif' in filename and folderName == path0:
            files.append(folderName+'/'+filename)
print('total files:', len(files))

#%% load transformation matrix from SIFT

tranM = []
MIN_MATCH_COUNT = 10
img1 = cv2.imread(files[0],0)  # Load an color image in grayscale
img2 = cv2.imread(files[1],0) # Load an color image in grayscale

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good) >= MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    tranM.append(M)

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    print('Image ',1,' and ',2, 'have an overlapped region')
else:
    print( "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None
    tranM.append(None)


#%% first image

im1 = cv2.imread(files[0])
im2 = cv2.imread(files[1])
ratio = height/im1.shape[0]
   
segfile = path0 + 'im0.seg1.pkl'
f = open(segfile,'rb')
seg1 = pickle.load(f)
f.close()
labels1 = edgeremove(seg1)

props = regionprops(seg1)
 
# second image
segfile = path0 + 'im1.seg1.pkl'
f = open(segfile,'rb')
seg2 = pickle.load(f)
f.close()
labels2 = edgeremove(seg2)

for label in labels1:
    
    # get bbox for different labels
    for i in range(len(props)):
        temp = props[i]
        if temp.label == label:
            prop1 = props[i]
            break
    
    [minR,minC,maxR,maxC] = prop1.bbox

    # get corresponding bbox on image 2
    pts = np.float32([ [minC,minR],[minC,maxR],[maxC,maxR],[maxC,minR] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,tranM[0])
    minR2 = min(int(round(min(dst[:,0,1]))),im2.shape[0]-1)
    maxR2 = min(int(round(max(dst[:,0,1]))),im2.shape[0]-1)
    minC2 = min(int(round(min(dst[:,0,0]))),im2.shape[1]-1)
    maxC2 = min(int(round(max(dst[:,0,0]))),im2.shape[1]-1)
    
    candidate = []
    if minR2 >= 0 and minC2 >= 0:
        bwp = seg2 == 9999
        for row in range(minR2,maxR2):
            for col in range(minC2,maxC2):
                bwp[row,col] = True
        target2 = list(np.unique(seg2[bwp]))
        if 0 in target2:
            target2.remove(0)
       
        if len(target2) > 0: # 表示在第二張圖有重疊的目標
            bw1 = seg1 == label
            col1 = int(0.5*maxC+0.5*minC)
            row1 = int(0.5*maxR+0.5*minR)
            col2 = int(0.5*maxC2+0.5*minC2)
            row2 = int(0.5*maxR2+0.5*minR2)
            halfc = int(0.5*min(maxC-minC,maxC2-minC2))
            halfr = int(0.5*min(maxR-minR,maxR2-minR2))
            
            crop1 = bw1[row1-halfr:row1+halfr,col1-halfc:col1+halfc]
            match = 0
            for target in target2:
                if target in labels2:
                    bw2 = seg2 == target
                    crop2 = bw2[row2-halfr:row2+halfr,col2-halfc:col2+halfc]
                    area = np.sum(crop1&crop2)
                    if len(candidate) == 0:
                        candidate.append(target)
                        match = area
                    if len(candidate) > 0 and area > match:
                        candidate = []
                        candidate.append(target)
                        match = area
            
    if len(candidate) == 1:
        cv2.destroyAllWindows()
        bw1 = seg1 == label
        bw2 = seg2 == candidate[0]
        bwim2(bw1,im1,'1',ratio)
        bwim2(bw2,im2,'2',ratio)
        cv2.waitKey(0)

cv2.destroyAllWindows()
