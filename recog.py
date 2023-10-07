import os

import cv2
import numpy as np
import torch
from glob import glob
import device
import infernance
def Prediction(img):
    path = glob('/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/models/')
    #path = os.path.join(path,)
    #print('Path : ',path)
    model = torch.load("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/models/best_model.pt")
    print("Prediction for one image")

    image = cv2.imread(img, cv2.IMREAD_COLOR)
    #print("IMAGE : ", type(image))
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    a = []
    i = 0
    #TODO : récuperer chaque image correspondant au découpage de la BB
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]



        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (18, 18))

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((7, 7), (7, 7)), "constant", constant_values=0)
        print("padded_digit.size() : ",padded_digit.size)


        #digit = padded_digit.reshape(1, 28, 28, 1)
        digit = padded_digit.reshape(1,  32, 32)
        print("digit.size() : ", digit.size)

        digit = torch.from_numpy(digit / 255.0)
        a.append(digit)
        if not os.path.isdir("images"):
            os.mkdir("images")
        cv2.imwrite("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/images/image_" + str(i) + ".png", padded_digit)

        i += 1
        #print('len digit : ', len(digit))
        #print("digit : ", digit)
        #res = model([digit][0].float()).detach()

        #prediction = torch.argmax(res, dim=-1)
        prediction = infernance.predict_image([digit][0].float(), model)
        #prediction = prediction.argmax().numpy()
        print("prediction =", prediction)
        # data = str(prediction) + ' ' + str(int(max(res) * 100)) + '%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, str(prediction), (x, y - 5), font, fontScale, color, thickness)

    #print(len(a))
    '''for i in a:
        print(i.shape)
        res = model([i][0].float()).detach()
        print("res : ",res)
        prediction = torch.argmax(res, dim=-1)
        print("prediction =", prediction)
        #prediction = res.argmax().numpy()
        #print("prediction =", prediction)
        #data = str(prediction) + ' ' + str(int(max(res) * 100)) + '%'
'''
    '''font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255, 0, 0)
    thickness = 1
    cv2.putText(image, str(prediction), (x, y - 5), font, fontScale, color, thickness)
'''
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
    #return np.argmax(res), max(res)'''