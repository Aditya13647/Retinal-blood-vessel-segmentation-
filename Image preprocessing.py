import cv2
import numpy as np
from glob import glob
import os
def load_data(path):
  x =  sorted(glob(os.path.join(path,"*.jpg")))
  return x

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def pre_procesing(paths):
        
        preprocess_imagepath = create_dir('D:mini project/hr/preproccessed')
        for i in range(len(paths)):       

         
                name  = (paths[i]).split('\\')[-1].split(".")[0]                   
                x = cv2.imread(paths[i],cv2.IMREAD_COLOR)
                x= cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
                # cv2.imshow("x",x)
                normalized_image_x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
                # cv2.imshow("normalized_image_x",normalized_image_x) 
                #rgb to lab conversion
                img_lab_x = cv2.cvtColor(normalized_image_x,cv2.COLOR_RGB2LAB)
                # cv2.imshow("rgb2lab",img_lab_x)         #cv2.imwrite('D:/mijnproject/PREPROCEESED/PreImage_01L.jpg',img_lab)       
                #L channel extraction
                lchannel_x,a_x,b_x = cv2.split(img_lab_x) 
                # cv2.imshow("Lchannel",lchannel_x) ##img_lab[:,:,0]#         #cv2.imwrite('D:/mijnproject/PREPROCEESED/PreImageLchannel_01L.jpg',lchannel)
                #CLAHE algorithm
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                # cv2.imshow("clahe",clahe) 
                clahe_img_x = clahe.apply(lchannel_x)
#         #Recombination of L channel with a and b componenets   
                # cv2.imshow("normalized_image_x",clahe_img_x)          
                lback = cv2.merge((clahe_img_x,a_x,b_x))
                # cv2.imshow("normalized_image_x",lback) 
                rgb_x = cv2.cvtColor(lback,cv2.COLOR_Lab2RGB)
                # cv2.imshow("lab",rgb_x) 
#         #extraction of green channel
                green_channel_x = rgb_x[:,:,1]
#         #Apply clahe algorithm
                contrast_img_x = clahe.apply(green_channel_x)
                # cv2.imshow("contrastimage",contrast_img_x) 
                tmp_image_name = f"{name}.tiff"
                image_path = os.path.join("D:\\mini project\\hr\\hr_dataset\\preproccessed\\",tmp_image_name)
                cv2.imwrite(image_path,contrast_img_x)

  
image_paths = load_data("D:\\mini project\\hr\\hr_dataset\\fundus_images")
print(len(image_paths))
pre_procesing(image_paths)
cv2.waitKey(0)
cv2.destroyAllWindows()
