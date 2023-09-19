 ####################################################               training         code                         ####################################################################

from keras.backend import binary_crossentropy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from metrics import sensitivity,specificity,auc
from staircase_netmodel import staircase_net


# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(path):
 
    y = sorted(glob(os.path.join(path,"*.jpg")))
    return y


    
def shuffling(x,y):
  #shuflle training image in the mask
   x,y = shuffle(x, y, random_state = 42)
   return x,y

def read_image(path):
    #path = path.tobytes()
    # print(type(path))
    path = path.decode()
    x = cv2.imread(path,cv2.IMREAD_COLOR)
    x = x/255.0 #normalizing
    x = x.astype(np.float32)
    return x

def read_mask(path):
    #path = path.tobytes()
    path = path.decode()
    x = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x,y) :
    def _parse(x,y):
        #x = np.array([file_path.encode() for file_path in x]) # convert to bytes
        #y = np.array([file_path.encode() for file_path in y]) # convert to bytes
        x = read_image(x)
        y = read_mask(y)
        return x,y
    x,y = tf.numpy_function(_parse, [x,y], [tf.float32, tf.float32])
    x.set_shape([H,W,3])
    y.set_shape([H,W,1])
    return x,y

def tf_dataset(X,Y, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset
    
  
         
   #Setting
np.random.seed(42)
tf.random.set_seed(42)
   
    #directory to save files
   
create_dir("files")
   
    #hyperparameters
batch_size = 2
lr = 1e-4
num_epochs = 60

model_path = os.path.join("C:/Users/shyam/OneDrive/Desktop/soft computing project","model.h5")
csv_path =os.path.join("C:/Users/shyam/OneDrive/Desktop/soft computing project","data.csv") 
   
    #dataset   
    #path = D:\mini project\New DATASET\a_data\train
train_y = load_data("C:/Users/shyam/OneDrive/Desktop/soft computing project/New DATASET/a_data/train/mask")
train_x = load_data("C:/Users/shyam/OneDrive/Desktop/soft computing project/New DATASET/a_data/preprocess/images") 
valid_x = load_data("C:/Users/shyam/OneDrive/Desktop/soft computing project/New DATASET/a_data/preprocess/mask") 
valid_y = load_data("C:/Users/shyam/OneDrive/Desktop/soft computing project/New DATASET/a_data/test/mask")

# train_x,valid_x = pre_procesing(train_x,valid_x)
train_x,train_y = shuffling(train_x,train_y)
   
   
print(f"Train:{len(train_x)} - {len(train_y)}")
print(f"Valid:{len(valid_x)} - {len(valid_y)}")
train_dataset = tf_dataset(train_x,train_y,batch_size=batch_size)
valid_dataset = tf_dataset(valid_x,valid_y,batch_size=batch_size)
  
train_Step = len(train_x)//batch_size
valid_Step = len(valid_x)//batch_size
   
if(len(train_x) % batch_size)!=0:
  train_Step += 1
if(len(valid_x) % batch_size)!=0:
  valid_Step += 1    
     
    #model
model = staircase_net((H,W,3))    


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'binary_accuracy', 'binary_crossentropy',sensitivity,specificity,auc])

# Define the callbacks
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# Train the model
history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        steps_per_epoch=train_Step,
        validation_steps=valid_Step,
        callbacks=callbacks
    )

print("callbacks")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("/content/drive/MyDrive/Colab Notebooks/newloss.png")

