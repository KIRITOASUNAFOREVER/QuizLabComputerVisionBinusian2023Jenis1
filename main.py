import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def gaussianBlur(img):
    gaussian = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT) 
    return gaussian

Target=cv2.imread("Dataset/Monument.jpg",0)

listImage=[]
for imagePath in os.listdir("Dataset/Data"):
    if imagePath.split('.')[1]=="jpg":
        listImage.append(cv2.imread("Dataset/Data/"+imagePath,0))

arrayMatches = []
for i in range(len(listImage)):
    image=[listImage[i]]
    listImage[i]=gaussianBlur(listImage[i])
    img_object = cv2.imread("Dataset/Monument.jpg")
    surf = cv2.xfeatures2d.SURF_create()
    surf.setHessianThreshold(400)
    kp_object, des_object = surf.detectAndCompute(img_object,None)
    kp_scene,des_scene = surf.detectAndCompute(listImage[i],None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE)
    search_params = dict(check=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matchess = flann.knnMatch(des_object,des_scene,k=2)
    matchesMasks = []
    totalMatches = 0
    for i in range(len(matchess)):
        matchesMasks.append([0, 0])
    for i, (m,n) in enumerate(matchess):
        if m.distance < 0.7 * n.distance:
            matchesMasks[i] = [1, 0]
            totalMatches += 1
    arrayMatches.insert(i,totalMatches)


maximal = 0
indexMaximal = -1
for j in range(len(arrayMatches)):
    if(arrayMatches[j] > maximal):
        maximal = arrayMatches[j]
        indexMaximal = j

surf = cv2.xfeatures2d.SURF_create()

img_object = cv2.imread("Dataset/Monument.jpg",0)
kp_object, des_object = surf.detectAndCompute(img_object,None)
kp_scene,des_scene = surf.detectAndCompute(listImage[indexMaximal],None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE)
search_params = dict(check=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des_object,des_scene,k=2)
matchesMask = []

for i in range(len(matches)):
    matchesMask.append([0, 0])

for i, (z,q) in enumerate(matches):
    if z.distance < 0.7 * q.distance:
        matchesMask[i] = [1, 0]

img_res = cv2.drawMatchesKnn(
    img_object, kp_object,
    listImage[indexMaximal],kp_scene,
    matches, None,
    matchColor=[0, 255, 0], #HIJAU
    singlePointColor=[0, 0 , 149], #BIRU
    matchesMask= matchesMask
)

plt.imshow(img_res)
plt.xticks([]), plt.yticks([])
plt.show()