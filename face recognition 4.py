#import necessary packages
import cv2
import numpy as np
import face_recognition
import os
import pickle

#load the numpy files
Data=np.load('encodeList.npy')
Target=np.load('target.npy')

#lode the dictonary
dict_file= open("frDict.pkl", "rb")
fr_dict = pickle.load(dict_file)
print(fr_dict)

final_list=[]

#read new image
img= cv2.imread('elon3.jpg')
#img = cv2.resize(img, (400, 400))
imgs = cv2.resize(img, (0, 0), None, 0.25, 0.25)
imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

#encode new image
locaNewface= face_recognition.face_locations(imgs)
encodeNewface = face_recognition.face_encodings(imgs, locaNewface)

#compare new image with pre encoded file
for encodeFace, faceLoc in zip(encodeNewface, locaNewface):
    matches = face_recognition.compare_faces(Data, encodeFace,tolerance=0.5)
    faceDis = face_recognition.face_distance(Data, encodeFace)
    print(matches)
    print(faceDis)

    #matchIndex = np.argmin(faceDis)
    #print(matchIndex)

# list of matched files
count=0
try:
    for i in range(0,len(matches)):
        if matches[i] == True:
            name = Target[i]
            print(name)
            #extracting the key by giving values from a dictonary
            output = [number for number, id in fr_dict.items() if id == name]
            final_list.append(output)
            print(output)
            count=count+1
except Exception as e:
    print("Please enter single pic")
    exit()

# create a final list by deleting the duplicate elements
final_output=[]
[final_output.append(x) for x in final_list if x not in final_output]



print(final_output)
if count==0:
    y1, x2, y2, x1 = faceLoc
    # print(x1, x2, y1, y2)
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "No Result Found", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
for final in final_output:
    y1, x2, y2, x1 = faceLoc
        #print(x1, x2, y1, y2)
    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.putText(img,str(final), (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
#print(matchIndex)

# if matches[matchIndex]:
#     name = Target[matchIndex]
#     print(name)
#     output = [number for number, id in fr_dict.items() if id == name]
#     print(output)
#     y1, x2, y2, x1 = faceLoc
#     #print(x1, x2, y1, y2)
#       y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
#     cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 0), cv2.FILLED)
#      cv2.putText(img,str(output), (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#  cv2.imshow('img', img)
#  cv2.waitKey(0)

dict_file.close()