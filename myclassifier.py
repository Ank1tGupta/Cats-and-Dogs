import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from PIL import Image

pick=open('data.pickle','rb')
data=pickle.load(pick)
pick.close()

random.shuffle(data)

features=[]
labels=[]
for feature,label in data:
    features.append(feature)
    labels.append(label)

X_train, X_test, y_train, y_test =train_test_split(features,labels,test_size=0.01)

pick=open('model.sav','rb')
model=pickle.load(pick)
pick.close()

prediction=model.predict(X_test)
accuracy=model.score(X_test,y_test)
print('Accuracy:',accuracy)

categories=['cats','dogs']
print('Prediction is :',categories[prediction[0]])

mypet=X_test[0].reshape(50,50)
plt.imshow(mypet,cmap='gray')
plt.show()