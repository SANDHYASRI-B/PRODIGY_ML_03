import pandas as pd
import os
from google.colab.patches import cv2_imshow
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pickle

#Load the cats_Dogs training dataset from google drive
os.chdir('/content/drive/MyDrive/cat__dog/dataset/training_set')

category=['cat','dog']
flat_data_arr=[] #input array
target_arr=[] #output array
for j in category:
  path=os.path.join('/content/drive/MyDrive/cat__dog/dataset/training_set',j)
  for img in os.listdir(path):
    img_array=imread(os.path.join(path,img))
    img_resized=resize(img_array,(150,150,3))
    flat_data_arr.append(img_resized.flatten())
    target_arr.append(category.index(j))
  print(f'Loaded category:{j} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)

#dataframe
df=pd.DataFrame(flat_data)
df['Target']=target
df.shape

#input data
x=df.iloc[:,:-1]
#output data
y=df.iloc[:,-1]

# Split the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15, random_state=0)

# Define the parameters grid
param_grid={'C':[100], 'gamma':[0.1], 'kernel':['poly']}

# Create a support vector classifier
svc=svm.SVC(probability=True)

# Create a model using GridSearchCV with the parameters grid
model=GridSearchCV(svc,param_grid)

# Train the model using the training data
model.fit(x_train,y_train)

# Test the model using the testing data
prediction = model.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(prediction, y_test)
print(f"The model is {accuracy*100}% accurate")

print(classification_report(y_test, prediction, target_names=['cat','dog']))

path='/content/drive/MyDrive/cat__dog/dataset/training_set/cat/cat.2332.jpg'
img=imread(path)
plt.imshow(img)
plt.show()
img_resize=resize(img,(150,150,3))
l=[img_resize.flatten()]
probability=model.predict_proba(l)

for ind,val in enumerate(category):
  print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+category[model.predict(l)[0]].title())
