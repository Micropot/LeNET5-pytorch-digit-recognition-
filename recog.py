import cv2
import numpy as np
import torch
from glob import glob
import os
def Prediction(img):
    path = glob('/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/models/')
    #path = os.path.join(path,)
    #print('Path : ',path)
    model = torch.load("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/models/mnist_0.891.pt")
    print("Prediction for one image")

    image = cv2.imread(img, cv2.IMREAD_COLOR)
    #print("IMAGE : ", type(image))
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    a = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]

        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = torch.from_numpy(digit / 255.0)
        a.append(digit)
        print(type(digit))
        print([digit][0].shape)
    print(len(a))
        #res = model([digit][0].float())
        #print("res : ", res)

    #res = model([digit])[0]
    #print("resut : ", res)
    #prediction = res.argmax()
    #print("prediction =", prediction)
    #data = str(prediction) + ' ' + str(int(max(res) * 100)) + '%'

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    #cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
    #return np.argmax(res), max(res)