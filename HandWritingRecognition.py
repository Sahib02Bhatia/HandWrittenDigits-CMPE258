#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 20:50:22 2022

@author: sahibbhatia
"""
#convert image to grayscale 
def convertToGrayScale(img):
    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_img = cv2.GaussianBlur(g_img, (5, 5), 0)
    g_img = cv2.adaptiveThreshold(g_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize = 321, C = 28)
    return g_img

def processImage(cap,j,Mnist_model,out):
    j=j+1
    ret, frame = cap.read()
    if ret == True: 
      img = cv2.resize(frame, (256,256))
      g_img = convertToGrayScale(img)
      edges = cv2.Canny(g_img, 100, 200)
      im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      predict(contours,Mnist_model,g_img,img,j)
      out.write(img)
      cv2.imshow('digits',img)   

def predict(contours,Mnist_model,g_img,img,j):
    
    for i in range(len(contours)):
        [x, y, w, h] = cv2.boundingRect(contours[i])
        if(h>10):
            pad =15
            img_original = copy(img)
            crop_image = g_img[y-pad:y+h+pad, x-pad:x+w+pad]
            crop_image =crop_image/255.0
            frame_name =  "frame_number_" + str(i)
          # cv2.imshow('digit', crop_image)
            try:
                crop_image = cv2.resize(crop_image, (28,28))
                pred = Mnist_model.predict(crop_image.reshape(1,28, 28, 1)).argmax()
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255.0, 0), 1)
                print(pred)
                prediction = str(pred) 
                cv2.putText(img, prediction, (x, y-20),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 0),1)   
                cv2.imshow(prediction, cv2.resize(img_original[y-pad:y+h+4, x-pad:x+w+4], (130,130)))
            except Exception as e:
                print(str(e))
                     
    return img


def main():
   
    print("Please choose -> Type: 1 for input video , Type: 2 for live video")   
    user_option = input()
   
    if(user_option == "1"):
        j=1    

        # Create a VideoCapture object and read from input file
        # If the input is taken from the camera, pass 0 instead of the video file name.
        cap = cv2.VideoCapture('sid.mp4')
    
        # Check if camera opened successfully
        if (cap.isOpened()== False):
          print("Error opening video File!!")
    
        # The default resolutions are system dependent which is obtained here
        # We convert the resolutions to integer. 
        target_size =(256,256)
        # Define codec and create VideoWriter object
        out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5 , target_size)
        start_time = time.time()
        Mnist_model = load_model('model.h5')
        while(cap.isOpened() and time.time()-start_time < 20):
            processImage(cap,j,Mnist_model,out)

            if cv2.waitKey(20) & 0xFF == ord('q'):
              break   

    else:
        j=1                                                                                   
        print("Inside the live_video!!!")
        cap = cv2.VideoCapture(0)
    
        if (cap.isOpened() == False): 
            print("Unable to read camera!!!")
    
        Mnist_model = load_model('model.h5')
        out = cv2.VideoWriter('Live_output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (256,256))
        start_time = time.time()
    
        while(int(time.time()-start_time) < 200):
            j=j+1
            ret, frame = cap.read()        
            if ret == True:
                try:
                    img = cv2.resize(frame, (256,256))
                    g_img = convertToGrayScale(img)
                    edges = cv2.Canny(g_img, 100, 200)
                    im2, contours , hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    predict(contours,Mnist_model,g_img,img,j)
                    #cv2.imshow(img)
                    #cv2.imshow(img)
                    out.write(img)
                    if cv2.waitKey(200) or 0xFF == ord('q') :
                        break
                except Exception as e:
                    print(str(e))
                

       

if __name__ == '__main__':
    import cv2
    from tensorflow.keras.models import load_model
    import time
    from google.colab.patches import cv2_imshow
    from copy import copy
    main()
    