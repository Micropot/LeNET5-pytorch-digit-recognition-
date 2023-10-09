import os
import cv2
import numpy as np
import torch
import infernance
from PIL import Image
from matplotlib import cm
import torchvision.transforms as transforms

def Prediction(img):
    model = torch.load("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/models/best_model.pt")
    print("Prediction for one image")

    image = cv2.imread(img, cv2.IMREAD_COLOR)
    print("IMAGE : ", type(image))
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    '''cv2.imshow('image', th)
    cv2.waitKey(0)
    cv2.destroyWindow('image')'''


    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    i = 0
    #TODO : récuperer chaque image correspondant au découpage de la BB
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # make a rectangle box around each curve
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Cropping out the digit from the image corresponding to the current contours in the for loop
        digit = th[y:y + h, x:x + w]
        im = Image.fromarray(np.uint8(cm.gist_earth(digit) * 255)) # convert array to PIL image
        w,h = im.size
        difference = np.abs(w-h)
        if w<h:
            padding = transforms.Pad((int(difference/2), 0))
        elif h<w:
            padding = transforms.Pad((0, int(difference/2)))
        else:
            padding = transforms.Pad(10)


        if not os.path.isdir("images_raw"):
            os.mkdir("images_raw")
        cv2.imwrite("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/images_raw/image_" + str(i) + ".png",
                    digit)

        trans = transforms.Compose([
            # To resize image
            # transforms.RandAugment(2, 9),
            padding,
            transforms.Pad((int(w*0.33),int(h*0.33))),
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),

            # To normalize image
            transforms.Normalize((0.5,), (0.5,))
        ])

        # apply transformation
        digit = trans(im)
        toPil = transforms.ToPILImage()
        new_image = toPil(digit)
        if not os.path.isdir("images_resized"):
            os.mkdir("images_resized")
        new_image.save("/Users/arthurlamard/Documents/ISEN5/deep_learning/CNN/TP1/images_resized/image_" + str(i) + ".png")
        print(digit.size())

        i += 1

        prediction = infernance.predict_image([digit][0], model)
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

