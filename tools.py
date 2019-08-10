import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from yolo import YOLO
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def read_img_boxes(boxStr,Img,allowed=['chair','person'] ,n=4):
    txt=boxStr.split('|')
    name=txt[0]
    boxes=txt[1:]
    if Img!=None:
        if name!=Img:
            return None,None
    l=[]
    names=[]
    for box in boxes:
        res=box.split(',')
        if res[0] not in allowed:
            continue
        b=res[-n:]
        l.append(b)
        names.append(res[0])
    return l,names

def iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def read_boxes(fileName,Img=None,n=4):
    with open(fileName) as file:
        imgStrs=file.read().split('\n')
        allBoxes=[]
        allNames=[]
        for imgStr in imgStrs:
            res,names=read_img_boxes(imgStr,Img,n=n)
            if res!=None and res!=[]:
                allBoxes.append(res)
                allNames.append(names)
        return allBoxes,allNames
        # return np.array(allBoxes).astype(float),np.array(allNames)


def cluster_crowd(img,features,boxes_,thickness,draw,colors,min_samples,eps,name):
    features = (features - np.mean(features,axis=0)) / np.std(features,axis=0)
    clusters = DBSCAN(min_samples=min_samples, eps=eps).fit(np.array(features)).labels_
    boxes_ = np.array(boxes_)
    u_clusters = np.unique(clusters[clusters != -1])
    for c in u_clusters:
        ind = np.where(clusters == c)[0]
        top, left = np.min(boxes_[ind, :2], axis=0)
        bottom, right = np.max(boxes_[ind, 2:], axis=0)
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
    cv2.imwrite(name,np.array(img))


# def cluster_crowd(img,features,boxes_,thickness,draw,colors,min_samples,eps,name):
#     features = (features - np.mean(features,axis=0)) / np.std(features,axis=0)
#     clusters = DBSCAN(min_samples=min_samples, eps=eps).fit(np.array(features)).labels_
#     print("Clusters:", clusters)
#     boxes_ = np.array(boxes_)
#     u_clusters = np.unique(clusters[clusters != -1])
#     for c in u_clusters:
#         ind = np.where(clusters == c)[0]
#         top, left = np.min(boxes_[ind, :2], axis=0)
#         bottom, right = np.max(boxes_[ind, 2:], axis=0)
#         for i in range(thickness):
#             draw.rectangle(
#                 [left + i, top + i, right - i, bottom - i],
#                 outline=colors[c])
#     cv2.imwrite(name,np.array(img))


def visualize(img,boxes,names=None,vis=True):
    colors = [(255, 255, 255), (225, 225, 0)]
    if names is not None:
        map=pd.Series(colors[:len(names)], np.unique(names))
    for i,box in enumerate(boxes):
        top, left, bottom, right=box.astype(int)
        if names is not None:
            cv2.rectangle(img,(left,top),(right,bottom),map[names[i]],2)
        else:
            cv2.rectangle(img, (left, top), (right, bottom), (255,255,255), 2)
    if vis:
        plt.subplots(figsize=(10, 10))
        plt.imshow(img[...,::-1])
        plt.show()
    return img

def detectVideoCrowd(video,allBoxes,allFeatures,allnames):
    res_video=[]
    for i in range(len(allBoxes)):
        features,boxes=allFeatures[i],allBoxes[i]
        boxes=np.array(boxes).astype(float)
        features=np.array(features).astype(float)
        names=np.array(allnames[i])
        per_features=features[names == 'person']
        per_boxes=boxes[names == 'person']
        res=[]
        if (len(per_boxes)):
            crowd=cluster_crowd(per_features,per_boxes,2,eps=0.9)
            res=visualize(video[i].copy(),crowd,vis=False)
            res_video.append(res)
        else:
            res_video.append(video[i].copy())
    return res_video

def cluster_crowd(features,boxes,min_samples,eps):
    features = (features - np.mean(features,axis=0)) / (np.std(features,axis=0)+np.spacing(1))
    clusters = DBSCAN(min_samples=min_samples, eps=eps).fit(np.array(features)).labels_
    u_clusters = np.unique(clusters[clusters != -1])
    crowd=[]
    for c in u_clusters:
        ind = np.where(clusters == c)[0]
        top, left = np.min(boxes[ind, :2], axis=0)
        bottom, right = np.max(boxes[ind, 2:], axis=0)
        crowd.append([top,left,bottom,right])
    return np.array(crowd)


class Cap:
    def __init__(self, path, step_size=0, reshape_size=(512, 512)):
        self.path = path
        self.step_size = step_size
        self.curr_frame_no = 0
        self.video_finished = False
        self.reshape_size = reshape_size

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.path)
        return self

    def read(self):
        success, frame = self.cap.read()
        if not success:
            self.video_finished = True
            return success, frame
        for _ in range(self.step_size - 1):
            s, f = self.cap.read()
            if not s:
                self.video_finished = True
                break
        return success, frame

    def read_all(self):
        frames_list = []
        while not self.video_finished:
            success, frame = self.read()
            if success:
                # frame = resize(frame , self.reshape_size ) * 255).astype(np.uint8)
                frames_list.append(frame)

        return frames_list

    def __exit__(self, a, b, c):
        self.cap.release()
        cv2.destroyAllWindows()

class Data:
    def __init__(self,configs):
        self.dataPath = configs['source']
        self.step=configs['step']
        self.size=configs['size']
        self.names=None
        if (configs['video']):
            with Cap(self.dataPath,self.step) as cap:
                self.data = cap.read_all()
        else:
            l=[]
            images=os.listdir(self.dataPath)
            self.names=images
            for i in images:
                l.append(cv2.imread(os.path.join(self.dataPath,i)))
            self.data=l
        self.resize()

    def resize(self):
        l=[]
        for i in self.data:
            l.append((resize(i,self.size)*255).astype(np.uint8) )
        self.data=np.array(l)

def save_video(images,path,fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height,width=images[0].shape[:2]
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for image in images:
        out.write(image)
    out.release()
    cv2.destroyAllWindows()