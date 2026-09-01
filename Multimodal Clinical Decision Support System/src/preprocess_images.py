import os
import cv2
import numpy as np 

def resize_with_padding(img,target_size=(512,512)):

    h,w = img.shape[:2]
    th,tw = target_size

    scale = min(tw/w,th/h)
    nw,nh = int(w*scale),int(h*scale)

    resized = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)

    pad_w = tw - nw
    pad_h = th - nh

    top = pad_h // 2
    bottom = pad_h -top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized,top,bottom,left,right,cv2.BORDER_CONSTANT,value=0)
    return padded

def preprocess_single_image(img_path,output_path,target_size=(512,512)):

    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image : {img_path}")

    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    img_processed = resize_with_padding(img_clahe,target_size)

    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    cv2.imwrite(output_path,img_processed)