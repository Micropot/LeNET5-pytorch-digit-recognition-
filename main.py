from train import training
from recog import Prediction
from upload import upload

def main():

    #train = training()
    img = upload()
    Prediction(str(img))





if __name__ == '__main__':
    main()