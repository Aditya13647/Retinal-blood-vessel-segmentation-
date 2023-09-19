from tkinter import filedialog
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
import customtkinter
from metrics import sensitivity,specificity,auc


filepath = None
image_path = None
save_path = None
inverted_path = None
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")


app = customtkinter.CTk()
app.geometry("600x300")
app.title("BLOOD VESSEL SEGMENTATION")



with CustomObjectScope({'sensitivity': sensitivity, 'specificity': specificity, 'auc': auc}):
    model = load_model("C:/Users/shyam/OneDrive/Desktop/soft computing project/model.h5")
    
def pre_procesing(filepath):
    name = (filepath).split('/')[-1].split(".")[0]
    x = cv2.imread(filepath, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (512, 512), interpolation=cv2.INTER_AREA)
    normalized_image_x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    # rgb to lab conversion
    img_lab_x = cv2.cvtColor(normalized_image_x, cv2.COLOR_RGB2LAB)
    # L channel extraction
    lchannel_x, a_x, b_x = cv2.split(img_lab_x)
    # CLAHE algorithm
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img_x = clahe.apply(lchannel_x)
    #Recombination of L channel with a and b componenets
    lback = cv2.merge((clahe_img_x, a_x, b_x))
    rgb_x = cv2.cvtColor(lback, cv2.COLOR_Lab2RGB)
    #extraction of green channel
    green_channel_x = rgb_x[:, :, 1]
    #Apply clahe algorithm
    contrast_img_x = clahe.apply(green_channel_x)
    tmp_image_name = f"{name}.jpg"
    global image_path
    image_path = os.path.join("C:/Users/shyam/OneDrive/Desktop/soft computing project/GUI ouputs", tmp_image_name)
    cv2.imwrite(image_path, contrast_img_x)
    cv2.imshow("preprocessed", contrast_img_x)   
    
def segmented(path):
    global save_path
    name = (path).split('/')[-1].split('.')[0]
    img = cv2.imread(image_path)
    # # Preprocess the image
    img = cv2.resize(img, (512, 512))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # Predict the mask
    mask = model.predict(img)
    # Convert the mask to image format
    mask = np.squeeze(mask, axis=0)
    mask = (mask > 0.5).astype(np.uint8) * 255
    # Save the mask if save_path is provided
    # if save_path is not None:
    tmp_image_name = f"{name}.jpg"
    save_path = os.path.join("C:/Users/shyam/OneDrive/Desktop/soft computing project/GUI ouputs/", tmp_image_name)
    cv2.imwrite(save_path, mask)
    cv2.imshow("segmented image", mask)


def button_callback():
    global filepath
    filepath = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")])

    img = cv2.imread(filepath)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    cv2.imshow("selected image", img)
 
frame_1 = customtkinter.CTkFrame(master=app)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

label_1 = customtkinter.CTkLabel(text = " Segmented results",master=frame_1, justify=customtkinter.LEFT)
label_1.pack(pady=10, padx=10)

button_1 = customtkinter.CTkButton(
    master=frame_1, text="Select Image", command=button_callback)
button_1.pack(pady=10, padx=10)         
    
button_2 = customtkinter.CTkButton(
    text="Preprocessed", master=frame_1, command=lambda: pre_procesing(filepath))
button_2.pack(pady=10, padx=10)

button_3 = customtkinter.CTkButton(
    text="segmented image", master=frame_1, command=lambda: segmented(image_path))
button_3.pack(pady=10, padx=10)

def clear():
    cv2.destroyAllWindows()
    
button_5 = customtkinter.CTkButton(text="Clear",
                                   master=frame_1, command=clear)
button_5.pack(pady=10, padx=10)    
 
cv2.waitKey(0)
app.mainloop()    
