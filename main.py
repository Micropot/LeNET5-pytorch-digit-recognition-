from train import training
from recog import Prediction
from upload import upload
from PIL import Image
def main():
    #train = training()
    img = upload()
    #img = Image.open(img)
    print("img : ",img)
    Prediction(str(img))


if __name__ == '__main__':
    main()