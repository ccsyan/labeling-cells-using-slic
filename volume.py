# -*- coding: utf-8 -*-
"""
"""
import os
import numpy as np
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import cv2
import skimage
from scipy.ndimage.interpolation import rotate

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def insert(upper,lower):
    middle = upper & lower
    temp = np.random.rand(upper.shape[0],upper.shape[1])
    bw = temp >= 0.5
    middle[upper ^ lower] = bw[upper ^ lower] # ^ is XOR
    return middle

def bigsub(im3d,seg2,label,path):
    
    bigfilter = seg2 == label
    
    props = regionprops(seg2*bigfilter)
    prop = props[0]
    [up,left,down,right]=prop.bbox
    up-=5
    left-=5
    down+=5
    right+=5
    clone_filter = bigfilter[up:down,left:right]
    
    #%% 先收集每層 clone 的 color distrance，然後再二分法
    cdist_layer = []
    for layer in range(im3d.shape[0]):
        # local crop
        img = im3d[layer,up:down,left:right,:].copy()
        
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:,:,0] = img2[:,:,0] * clone_filter
        img[:,:,1] = img2[:,:,1] * clone_filter
        img[:,:,2] = img2[:,:,2] * clone_filter
        
        avgRGB_layer = np.mean(img[clone_filter],0)
        cdist = np.sqrt( avgRGB_layer[0]**2 + avgRGB_layer[1]**2 + avgRGB_layer[2]**2 )
        cdist_layer.append(cdist)
        
        
    signal_layer = skimage.filters.threshold_otsu(np.array(cdist_layer))
    
    take_layer = cdist_layer > signal_layer
    
    for i in range(len(take_layer)):
        if take_layer[i]:
            start_layer = i
            break
    
    for i in range(start_layer,len(take_layer)):
        if ~take_layer[i]:
            end_layer = i
            break
    
    
    #%%
    withcells = [] # 準備存放有 1 到 7 cell 的 seg
    first_peak = False
    accu_size = []
    for layer in range(start_layer,end_layer+1):
        top = 1
        # local crop
        img = im3d[layer,up:down,left:right,:].copy()
    
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:,:,0] = img2[:,:,0] * clone_filter
        img[:,:,1] = img2[:,:,1] * clone_filter
        img[:,:,2] = img2[:,:,2] * clone_filter
        
        # dynamical range rescale
        top = np.max(img)
        if top == 0: # 出現過這種情況
            top = 1
        base = np.min(img)
        img[:,:,:] = np.uint8( ( (img[:,:,:]-base) / (top - base ) *255 ) )
        # layer slic
        lseg0 = slic(img,
        #            start_label=1, # 在 python 3.8 label 沒有 0，從 1 開始
                n_segments = int(img.shape[0]*img.shape[1]/75), # 還是包含細胞邊
                compactness=.1, # 再小就沒差了 
                max_iter=50, # 再高也沒差      
                multichannel=True,
                convert2lab=True,
                enforce_connectivity=True,
                slic_zero=True,
                sigma = [1.0,1.0,1.0], # 0.1 只是讓邊緣變毛
                spacing=[1,1,1],#z,y,x # 縮小也沒幫助
                min_size_factor=0.4,
                max_size_factor=3)
    
        #%% dark cell threshold
        cdist = []
        lseg1 = lseg0.copy()
        lseg1[~clone_filter] = 0
        clabels = np.unique(lseg1)
        for clabel in clabels: # 這樣寫就不怕 label starts from 0 or from 1
            bw1 = lseg1 == clabel
            avgRGB = np.mean(img[bw1],0)
            # img[bw] 第一維度是 area，第二維度是 RGB，0 表示對第一維度取平均
            cdist1 =np.sqrt( avgRGB[0]**2 + avgRGB[1]**2 + avgRGB[2]**2 )
            cdist.append(cdist1)
        
        try: # 有時只有一個 sample 不能分
            dark_cell = skimage.filters.threshold_otsu(np.array(cdist))
            # 用大津法找出亮暗的 threshold
            # formatted_float = "{:.3f}".format(dark_cell)
            # print('dark threshold is',formatted_float)
        except:
            dark_cell = 0
#        print(dark_cell)
        #%% relabelling the cell label
        lseg3 = lseg1.copy()
        lseg3[:,:]=10
        
            
        lseg2 = lseg1.copy()
        
        lseg2[~clone_filter] = 0
        clabels = np.unique(lseg2)   
    #        print(labels)
        for clabel in clabels:
            if clabel > 0:
                bw1 = lseg2 == clabel
                avgRGB = np.mean(img[bw1],0) 
                # img[bw] 第一維度是 area，第二維度是 RGB，0 表示對第一維度取平均
                cdist1 =np.sqrt( avgRGB[0]**2 + avgRGB[1]**2 + avgRGB[2]**2 )
                
                if cdist1 < dark_cell:
                    lseg3[bw1] = 10 # 10是背景 label
                else:
                    lseg3[bw1] = 1
    
    #%% 開始輸出 manually picking results, 這邊沒有第幾個 layer 的資訊
        bw = lseg3 == 1 # 用 9 去掉不要的 segments
        accu_size.append(np.sum(bw))
        
        im2 = np.uint8(mark_boundaries(img, lseg3, color=(0, 0, 1))*255)
        img4 = cv2.resize(im2, (img.shape[1]*5, img.shape[0]*5), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(path+'label.'+str(label).zfill(3)+
                    '.layer.'+str(layer).zfill(2)+'.png',img4)
        withcells.append(lseg3.copy())

        if len(accu_size) > 6 and accu_size[-2] < accu_size[-1]:
            first_peak = True
        
        if first_peak and accu_size[-1] <  0.5*np.max(accu_size) + 0.5*np.min(accu_size):
            break
#%% 開始旋轉
    ctarget = 1
    layers = len(withcells)
    full_layers = (layers+1) * 4 + 1
    vseg1 = withcells[0]
    voxels = np.zeros((full_layers,vseg1.shape[0],vseg1.shape[1]),dtype='bool')
    
    for i in range(layers):
        vseg1 = withcells[i]
#        print(i,np.sum(vseg1 == ctarget))
    
    # 把每層向上的四層補進去
    layer_index=0
    for i in range(layers):
    
        vseg1 = withcells[i]
        
        if i == 0:
            layer0 = vseg1 > 999
        else:
            last_seg = withcells[i-1]
            layer0 = last_seg == ctarget
            
        layer4 = vseg1 == ctarget
        
        layer2 = insert(layer0,layer4)
        layer1 = insert(layer0,layer2)
        layer3 = insert(layer2,layer4)
            
        voxels[layer_index,:,:]=layer0
        voxels[layer_index+1,:,:]=layer1
        voxels[layer_index+2,:,:]=layer2
        voxels[layer_index+3,:,:]=layer3
            
        layer_index +=4
    
    # 處理最後一層
    layer4 = vseg1 > 999
    layer0 = withcells[-1] == ctarget
    
    layer2 = insert(layer0,layer4)
    layer1 = insert(layer0,layer2)
    layer3 = insert(layer2,layer4)
    
    voxels[layer_index,:,:]=layer0
    voxels[layer_index+1,:,:]=layer1
    voxels[layer_index+2,:,:]=layer2
    voxels[layer_index+3,:,:]=layer3
    
    #%% 以下在找 boundary box
    [zmax, rmax, cmax] = voxels.shape
    
    zstart = -1
    zend = -1
    for i in range(zmax):
        if np.sum(voxels[i,:,:]) > 0 and zstart == -1:
            zstart = i
            break
    for i in range(zstart,zmax):    
        if np.sum(voxels[i,:,:]) == 0:
            zend = i
            break
    
    rstart = -1
    rend = -1
    for i in range(rmax):
        if np.sum(voxels[:,i,:]) > 0 and rstart == -1:
            rstart = i
            break
    for i in range(rstart,rmax):
        if np.sum(voxels[:,i,:]) == 0:
            rend = i
            break
    
    cstart = -1
    cend = -1
    for i in range(cmax):
        if np.sum(voxels[:,:,i]) > 0 and cstart == -1:
            cstart = i
            break
    for i in range(cstart,cmax):
        if np.sum(voxels[:,:,i]) == 0:
            cend = i
            break    
    minvol = voxels[zstart:zend,rstart:rend,cstart:cend]
    #%% 轉成 xyz 後，轉角度比較符合直觀
    xyzvol = np.zeros((minvol.shape[2],minvol.shape[1],minvol.shape[0]), dtype=bool)
    
    for z in range(minvol.shape[0]):
        for r in range(minvol.shape[1]):
            for c in range(minvol.shape[2]):
                if minvol[z,r,c]:
                    xyzvol[c,minvol.shape[1]-r-1,minvol.shape[0]-z-1]=True
    
    #%% rotate xz plane to have max surface on xy plane
    if np.sum(xyzvol) > 0:
        maxsurface = 0
        good_xzangle = 0
        for xzangle in range(45,-46,-1):
            newvol = rotate(xyzvol, xzangle, axes=(0,2), reshape=True, order=0, # order 0 比 5 準
                            mode='constant') # constant is good
            surface = newvol[:,:,0]
        
            for z in range(1,newvol.shape[2]):
                surface = surface | newvol[:,:,z]
        
            if np.sum(surface) > maxsurface:
                maxsurface = np.sum(surface)
                good_xzangle = xzangle
        
        r2ndvol = rotate(xyzvol, good_xzangle, axes=(0,2), reshape=True, order=0, # order 0 比 5 準
                            mode='constant') # constant is good
        
        #%% rotate yz plane to have max surface on xy plane
        
        maxsurface = 0
        good_yzangle = 0
        for yzangle in range(45,-46,-1):
            newvol = rotate(r2ndvol, yzangle, axes=(1,2), reshape=True, order=0, # order 0 比 5 準
                            mode='constant') # constant is good
            surface = newvol[:,:,0]
        
            for z in range(1,newvol.shape[2]):
                surface = surface | newvol[:,:,z]
        
            if np.sum(surface) > maxsurface:
                maxsurface = np.sum(surface)
                good_yzangle = yzangle
        
        newvol = rotate(r2ndvol, good_yzangle, axes=(1,2), reshape=True, order=0, # order 0 比 5 準
                            mode='constant') # constant is good
        
        #%%
        thickness = np.uint8(newvol[:,:,0])
        for z in range(1,newvol.shape[2]):
            thickness += np.uint8(newvol[:,:,z])
        
        heights = np.unique(thickness)
        volume = 0
        yet = True
        for height in heights:
            amount = np.sum(thickness==height)
            volume += height * amount
            if yet and volume > 0.5* np.sum(newvol):
                cell_height = height
                yet = False
        # 直接轉成 micro-meter
        vol_micro_meter = (465/1024)**2 * 0.5
        
        val1 = np.sum(newvol)*vol_micro_meter
        
        val2 = maxsurface*vol_micro_meter**(2/3)
        
        val3 = cell_height*vol_micro_meter**(1/3)
    
    return([val1,val2,val3,good_xzangle,good_yzangle])
