import os

import cv2
import numpy as np
import torch
from glob import glob
import device
import infernance
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

def prepare_image(path: str):
    """
    Converting image to MNIST dataset format
    """

    im = Image.open(path).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        new_image.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        new_image.paste(img, (wleft, 4))  # paste resized image on white canvas

    pixels = list(new_image.getdata())  # get pixel values
    pixels_normalized = [(255 - x) * 1.0 / 255.0 for x in pixels]

    # Need adequate shape
    adequate_shape = np.reshape(pixels_normalized, (1, 28, 28))
    output = torch.FloatTensor(adequate_shape).unsqueeze(0)
    return output
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
    i = 0
    #TODO : récuperer chaque image correspondant au découpage de la BB
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]
        print('-------------  digit -------------')
        print(digit)
        if not os.path.isdir("images_raw"):
            os.mkdir("images_raw")
        cv2.imwrite("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/images_raw/image_" + str(i) + ".png",
                    digit)



        # Resizing that digit to (18, 18)
        resized_digit = cv2.resize(digit, (10, 10), interpolation=cv2.INTER_AREA)
        print('------------- resized digit -------------')
        print(resized_digit)

        # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
        padded_digit = np.pad(resized_digit, ((11, 11), (11, 11)), "constant")
        print('------------- padded digit -------------')
        print(padded_digit)
        #padded_digit = padded_digit.reshape(1, 32, 32)
        #padded_digit_torch = torch.from_numpy(padded_digit)



        #digit = padded_digit.reshape(1, 28, 28, 1)
        digit = padded_digit.reshape(1,  32, 32)
        print("digit.size() : ", digit)


        '''trans = transforms.Normalize((0.5,), (0.5,))
        digit = trans(digit.float())'''
        if not os.path.isdir("images"):
            os.mkdir("images")
        cv2.imwrite("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/images/image_" + str(i) + ".png", padded_digit)
        digit = torch.from_numpy(digit / 255.0)
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


    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyWindow('image')

