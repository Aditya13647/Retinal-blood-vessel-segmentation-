###########################################                        staircasenet model code                                      ######################################################=



from tensorflow.keras.layers import Conv2D,concatenate,Conv2DTranspose,Input,BatchNormalization,Activation,MaxPooling2D
from tensorflow.keras.models import Model

def staircase_net(input_shape):
    inputs = Input(shape = input_shape)
  
    #convolution block1
    conv1 = Conv2D(64,3,activation = 'relu',padding = 'same')(inputs)
    x = BatchNormalization()(conv1)    
    conv2 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv2)    
    conv3 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    op1 = BatchNormalization()(conv3)
    
    #upsampling layer 1 
    up1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")(op1)
    
    #convolution block 2
    conv4 = Conv2D(32,3,activation = 'relu',padding = 'same')(up1)
    x = BatchNormalization()(conv4)
    conv5 = Conv2D(32,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv5)
    conv6 = Conv2D(32,3,activation = 'relu',padding = 'same')(x)
    op2 = BatchNormalization()(conv5)
    
    #downsampling layer
    pool1 = MaxPooling2D(pool_size = (2,2),strides=(2,2))(op2)
    concat1 = concatenate([op1,pool1])
    
    #convolution block 3    
    conv7 = Conv2D(64,3,activation = 'relu',padding = 'same')(concat1)
    x = BatchNormalization()(conv7)
    conv8 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv8)
    conv9 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv9)
    conv10 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    op3 = BatchNormalization()(conv10)
    
    #downsampling
    pool2 = MaxPooling2D(pool_size = (2,2),strides=(2,2))(op3)
    
    #convolution block 4
    conv11 = Conv2D(128,3,activation = 'relu',padding = 'same')(pool2)
    x = BatchNormalization()(conv11)
    conv12 = Conv2D(128,3,activation = 'relu',padding = 'same')(x)                                                                 
    x = BatchNormalization()(conv12)
    conv13 = Conv2D(128,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv13)
    
    #upsampling
    up2 = Conv2DTranspose(128,3,strides=(2,2),activation = 'relu',padding = 'same')(x)
    concat2 = concatenate([op3,up2])
    
    #convolution block5    
    conv14 = Conv2D(16,3,activation = 'relu',padding = 'same')(concat2)
    x = BatchNormalization()(conv14)
    conv15 = Conv2D(16,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv15)
    conv16 = Conv2D(16,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv16)
   
    conv17 = Conv2D(1,1,activation = 'sigmoid',padding = 'same')(x)
    op4 = BatchNormalization()(conv17)
    
    concat3 = concatenate([inputs,conv17])
    
    #convolution block 6
    conv18  = Conv2D(64,3,activation = 'relu',padding = 'same')(concat3)
    x = BatchNormalization()(conv18)
    conv19 = Conv2D(64,3,activation = 'relu' ,padding = 'same')(conv18)
    x = BatchNormalization()(conv19)
    conv20 = Conv2D(64,3,activation = 'relu',padding='same')(conv19)
    op5 = BatchNormalization()(conv20)
    
    up3 = Conv2DTranspose(64,3,strides=(2,2),activation = 'relu',padding = 'same')(x)
    conv21 = Conv2D(32,3,activation = 'relu',padding = 'same')(up3)
    x = BatchNormalization()(conv21)
    conv22 = Conv2D(32,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv22)
    conv23 = Conv2D(32,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv23)
    pool3 = MaxPooling2D(pool_size = (2,2),strides=(2,2))(x)
    
    concat4 = concatenate([op5,pool3])

    conv24 = Conv2D(64,3,activation = 'relu',padding = 'same')(concat4)
    x = BatchNormalization()(conv24)
    conv25 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv25)
    conv26 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv26)
    conv27 = Conv2D(64,3,activation = 'relu',padding = 'same')(x)
    op6 = BatchNormalization()(conv27)   
    
    pool4 = MaxPooling2D(pool_size = (2,2),strides=(2,2))(op6)
   
    conv28 = Conv2D(128,3,activation = 'relu',padding = 'same')(pool4)
    x = BatchNormalization()(conv28)
    conv29 = Conv2D(128,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv29)
    conv30 = Conv2D(128,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv30)
  
    up4 = Conv2DTranspose(128,3,strides=(2,2),activation = 'relu',padding = 'same')(x)
    concat5 = concatenate([op6,up4])
    
    conv31 = Conv2D(16,3,activation = 'relu',padding = 'same')(concat5)
    x = BatchNormalization()(conv31)
    conv32 = Conv2D(16,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv32)
    conv33 = Conv2D(16,3,activation = 'relu',padding = 'same')(x)
    x = BatchNormalization()(conv33)
    
    conv34 = Conv2D(1,1,activation = 'sigmoid',padding = 'same')(x)
    
    
    model = Model(inputs, conv34, name="staircasenet")
    
    return model
 

input_shape = (512, 512, 3)
model = staircase_net(input_shape)
model.summary()
