import os
import numpy as np
import cv2
from glob import glob               #extraction of parts of images
from tqdm import tqdm               #progress bar
import imageio                      #to read gif file
from albumentations import GridDistortion,Rotate,ShiftScaleRotate,RandomGamma,RandomScale,GaussNoise,HorizontalFlip,VerticalFlip,ElasticTransform,CoarseDropout,OpticalDistortion

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# loading the images to the train_x and train_Y and test_x and test_y
# path = d --> mini project --> DATASET --> DRIVE

def load_data(path):
    #x = images , y =masks
    train_x = sorted(glob(os.path.join(path,"New DATASET","train","images","*.tif")))
    train_y = sorted(glob(os.path.join(path,"New DATASET","train","1st_manual","*.gif")))
    
    test_x = sorted(glob(os.path.join(path,"New DATASET","test","images","*.tif")))
    test_y = sorted(glob(os.path.join(path,"New DATASET","test","1st_manual","*.gif")))
    return (train_x,train_y),(test_x,test_y)


# augmenting only training data with various methods(augment = True)
# resizing the testing data (512*512)*(augment = False)

def data_agumentation(images, masks, save_path, agument = True):
    H = 512
    W = 512
    for idx, (x,y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = x.split("\\")[-1].split(".")[0]
        print(name)
        
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        
        if agument == True:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask= y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            
            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask= y)
            x2 = augmented["image"]
            y2 = augmented["mask"]
            
            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']
            
            aug = Rotate(limit=30)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']
          
               
            X = [x,x1,x2,x3,x4,x5,x6]
            Y = [y,y1,y2,y3,y4,y5,y6]
            
        else:
            X = [x]
            Y = [y]
        index = 0    
        for i, m in zip(X,Y):
            i = cv2.resize(i,(W,H))
            m = cv2.resize(m,(W,H))
            
            if( len(X) == 1 ):
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name}_{index}.jpg"    
            image_path = os.path.join(save_path,"image",tmp_image_name)
            mask_path = os.path.join(save_path,"mask", tmp_mask_name)
            
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            
            index += 1
        
    
if __name__ == "__main__":
    ##seeding
    np.random.seed(42)    
      
    #load the data
    data_path = "D:\mini project"  
    (train_x,train_y),(test_x,test_y) = load_data(data_path)
    
      
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    #create directories for agumentad data
    create_dir("D:mini project/New DATASET/a_data/train/image")
    create_dir("D:mini project/New DATASET/a_data/train/mask")
    create_dir("D:mini project/New DATASET/a_data/test/image")
    create_dir("D:mini project/New DATASET/a_Data/test/mask")
    
    data_agumentation(train_x, train_y, "D:mini project/New DATASET/a_data/train/",agument=True)
    data_agumentation(test_x,test_y,"D:mini project/New DATASET/a_data/test/",agument=False)
