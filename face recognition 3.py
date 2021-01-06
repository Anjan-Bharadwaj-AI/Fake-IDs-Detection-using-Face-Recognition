# importing all the necessary library
import cv2
import numpy as np
import face_recognition
import os
import pickle

# fatching images and the labels from folder
data_path='images'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels))

print(label_dict)
print(categories)
print(labels)
print(type(label_dict))

#list to store encodings and labels
encodeList=[]
target=[]

#pickle file to store the dictonary
dict_file = open("frDict.pkl", "wb")
pickle.dump(label_dict, dict_file)
dict_file.close()

# this loop is used for encoding of all images present inside the folders
for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            print('Encoding Complete')
            target.append(label_dict[category])
            #print(type(encodeList))
        except Exception as e:
            print("Same")

print(len(encodeList))
print(target)
print(type(encode))

#save the encode list and label in a numpy array file.
np.save('encodeList',encodeList)
np.save('target',target)


