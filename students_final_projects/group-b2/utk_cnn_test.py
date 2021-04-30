
import os



# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.

style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical

# specifically for cnn
from tensorflow.keras.layers import Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,LearningRateScheduler
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
import math
import seaborn as sn

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# X=[]
# Z=[]


save_history=[]
save_train_accuracy=[]
save_test_accuracy=[]
save_valid_accuracy=[]



save_confusion=[]

sid=['101','102','103','104','105','106']

input_directory=os.getcwd()
list_directory=os.listdir(input_directory)
list_directory= [i for i in list_directory if 'utk_cnn_test.py' not in i.lower()]
list_directory=['./' + l for l in list_directory]
print("\n List of directory found :{}".format(list_directory))
        

def assign_label(img,flower_type):
    return flower_type
    

count=0
height=0 
width=0
channels=0

def make_train_data(DIR,current):
    #### make a list of all sub directory####
    list_label=os.listdir(DIR)
    print(list_label)
    list_dir=[ DIR+'/'+l for l in list_label]
    i=0
    global count,height,width,channels
    for l in list_dir:
        for img in tqdm(os.listdir(l)):
            label=assign_label(img,list_label[i])
            path = os.path.join(l,img)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            
            if count==0:
                height, width, channels = img.shape
                print("\n height of image:{}".format(height))
                print("\n width of image:{}".format(width))
                print("\n channel of image:{}".format(channels))
                count=count+1
            
            ########## store in proper list####
            if DIR==current:
                x_test.append(np.array(img))
                y_test.append(str(label))
            else:
                x_train.append(np.array(img))
                y_train.append(str(label))
        i=i+1


#### for each subject id ########
count_participant=0
order_of_access=[]
for s in list_directory:    ## for each subject
    order_of_access.append(s)
    current_subject=s
    x_train=[]
    x_test=[]
    y_train=[]
    y_test=[]
    x_valid=[]
    y_valid=[]
    print("\n ******** Current Participant {} ***********!! \n".format(s))
    for d in list_directory: ## for each directory
        # remaining_subject=[i for i in list_directory if s not in i.lower()]
        print("\n loading {} directory.......!! \n".format(d))
        make_train_data(d,current_subject)
    
        #####



    fig,ax=plt.subplots(2,2)
    fig.set_size_inches(15,15)
    for i in range(2):
        for j in range (2):
            l=rn.randint(0,len(y_train))
            #ax[i,j].imshow(x_train[l])
            resized = cv2.resize(x_train[l], (100,height), interpolation = cv2.INTER_AREA)
            ax[i,j].imshow(resized)
            ax[i,j].set_title('Activity: '+y_train[l])  
    plt.tight_layout()
    plt.savefig('%s.png' % (s+'sample'))
        
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.75,test_size=0.25, random_state=42)

    #### fix the label ###################################
    
    print("\n train label found:{}".format(len(y_train)))
    print("\n test label found:{}".format(len(y_test)))
    print("\n valid label found:{}".format(len(y_valid)))
    le=LabelEncoder()
    y_train=le.fit_transform(y_train)
    y_train=to_categorical(y_train,6)
    
    y_test=le.fit_transform(y_test)
    y_test=to_categorical(y_test,6)
    
    
    y_valid=le.fit_transform(y_valid)
    y_valid=to_categorical(y_valid,6)
    
    y=le.classes_
    d=le.transform(y)
    print("label {} is represeneted {}".format(y,d))
    
    ##################################

    ########### fix the data########
    
    x_train=np.array(x_train)
    x_train=x_train/255
    
    
    x_test=np.array(x_test)
    x_test=x_test/255
    
    x_valid=np.array(x_valid)
    x_valid=x_valid/255
    
    print("\n train data found:{}".format(len(x_train)))
    print("\n test data found:{}".format(len(x_test)))
    print("\n valid data found:{}".format(len(x_valid)))
    ########################################################




    ######### Define the CNN model UCNet6################
    
    # model = Sequential()
    # model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (height, width, channels)))
    # model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.6))
    
    # model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    # model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.6))
    
    
    # model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    # model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    
    
    # model.add(Flatten())
    # model.add(BatchNormalization())
    # model.add(Dense(512))
    # model.add(Dropout(0.6))
    # model.add(Activation('relu'))
    
    
    # model.add(Dense(128))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    
    # model.add(Dense(6, activation = "softmax"))
    
    ####################################################
    
    ############### My model #############################
    
    
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (height, width, channels)))
    model.add(AveragePooling2D(pool_size=(2,2),strides=None, padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(AveragePooling2D(pool_size=(3,3),strides=None, padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(AveragePooling2D(pool_size=(2,2),strides=None, padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    
    # model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(Conv2D(filters = 512, kernel_size = (3,3),padding = 'Same',activation ='relu'))
    model.add(AveragePooling2D(pool_size=(3,3),strides=None, padding="same"))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Activation('relu'))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    
    
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(6, activation = "softmax"))
##########################################################################################

# x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

    import keras 
    class TestCallback (keras.callbacks.Callback):
        def __init__(self,test_data):
            self.test_data=test_data
            self.loss=[]
            self.acc=[]
        def on_epoch_end(self,epoch,logs={}):
            x,y=self.test_data
            loss,acc=self.model.evaluate(x,y,verbose=0)
            self.loss.append(loss)
            self.acc.append(acc)
            print('\n\nTesting loss: {}, acc: {}\n\n'.format(loss,acc))
    callback_test=TestCallback((x_test,y_test))

    np.random.seed(42)
    rn.seed(42)
    tf.random.set_seed(42)
    
    def step_decay(epoch):
       initial_lrate = 0.00002
       drop = 0.5
       epochs_drop = 10.0
       lrate = initial_lrate * math.pow(drop,  
               math.floor((1+epoch)/epochs_drop))
       return lrate
   ##### uncomment if you need learning rate scheduling save model and log
    lrs = LearningRateScheduler(step_decay)
    # ld=TensorBoard(log_dir='./logs')
    model_name=s+'_best_model.h5'
    mc = ModelCheckpoint(model_name, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)



    es = EarlyStopping(monitor='val_loss',min_delta=0,patience=20,verbose=0, mode='auto')




    batch_size=500
    epochs=80
    model.compile(optimizer=SGD(lr=0.001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()


    History = model.fit(x_train,y_train,epochs = epochs, validation_data = (x_valid,y_valid),verbose = 1,callbacks=[callback_test,es,lrs,mc,])
    save_history.append(History)
    
    

    scores_test=model.evaluate(x_test,y_test,verbose=0)
    save_test_accuracy.append(scores_test)
    print('Final epoch Testing Accuracy={}'.format(scores_test[1]))
    
    scores_test=model.evaluate(x_train,y_train,verbose=0)
    save_train_accuracy.append(scores_test)
    print('Final epoch Training Accuracy={}'.format(scores_test[1]))
    
    scores_test=model.evaluate(x_valid,y_valid,verbose=0)
    save_valid_accuracy.append(scores_test)
    print('Final epoch Validation Accuracy={}'.format(scores_test[1]))


    fig=plt.figure()
    plt.plot(History.history['accuracy'],"-b", label="Train-acc")
    plt.plot(History.history['val_accuracy'],"-r", label="valid-acc")
    plt.plot(callback_test.acc,"-g", label="test-acc")
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Metrics vs epoch')
    plt.legend(loc="lower right")
    plt.savefig('%s.png' % (s+'accuracy'))


    
    
    fig=plt.figure()
    plt.plot(History.history['loss'],"-b", label="Train-loss")
    plt.plot(History.history['val_loss'],"-r", label="valid-loss")
    plt.plot(callback_test.loss,"-g", label="test-loss")
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Loss Metrics vs epoch')
    plt.legend(loc="upper right")
    plt.savefig('%s.png' % (s+'loss'))

    

    test_predictions=model.predict(x_test)
    test_predictions=np.argmax(test_predictions,axis=1)
    rounded_labels=np.argmax(y_test,axis=1)
    cm=confusion_matrix(rounded_labels,test_predictions)
    class_sum=cm.sum(axis=1)
    diagonal=np.diagonal(cm)
    class_accuracy=diagonal/class_sum
    
    plt.figure(figsize = (10,7))
    sn.heatmap(cm, annot=True)
    plt.title("Confusion Matrix", fontsize =20)
    plt.savefig('%s.png' % (s+'confusion'))


##### calculation overall performance ####

save_train_accuracy=np.array(save_train_accuracy)
save_test_accuracy=np.array(save_test_accuracy)
save_valid_accuracy=np.array(save_valid_accuracy)

##### calculate average #################
avg_training_accuracy=np.average(save_train_accuracy[:,1])
avg_training_loss=np.average(save_train_accuracy[:,0])

avg_testing_accuracy=np.average(save_test_accuracy[:,1])
avg_testing_loss=np.average(save_test_accuracy[:,0])

avg_validation_accuracy=np.average(save_valid_accuracy[:,1])
avg_validation_loss=np.average(save_valid_accuracy[:,0])

################ plot the graph for individual #######

# fig=plt.figure()
fig, axs = plt.subplots(3, 1)
axs[0].bar(sid,save_test_accuracy[:,1])
axs[0].set_title('test')
axs[1].bar(sid,save_train_accuracy[:,1])
axs[1].set_title('train')
axs[2].bar(sid,save_valid_accuracy[:,1])
axs[2].set_title('valid')
for ax in axs.flat:
    ax.set(xlabel='Subject_ID', ylabel='Accuracy')

for ax in axs.flat:
    ax.label_outer()
plt.savefig('%s.png' % (s+'SID_accuracy'))


fig, axs = plt.subplots(3, 1)
axs[0].bar(sid,save_test_accuracy[:,0])
axs[0].set_title('test')
axs[1].bar(sid,save_train_accuracy[:,0])
axs[1].set_title('train')
axs[2].bar(sid,save_valid_accuracy[:,0])
axs[2].set_title('valid')
for ax in axs.flat:
    ax.set(xlabel='Subject_ID', ylabel='Loss')

for ax in axs.flat:
    ax.label_outer()
plt.savefig('%s.png' % (s+'SID_loss'))


print("\n avg_training_accuracy:{}".format(avg_training_accuracy))

print("\n avg_training_loss: {}".format(avg_training_loss))

print("\n avg_testing_accuracy:{}".format(avg_testing_accuracy))

print("\n avg_testing_loss:{}".format(avg_testing_loss))

print("\n avg_validation_accuracy:{}".format(avg_validation_accuracy))

print("\n avg_validation_loss:{}".format(avg_validation_loss))
