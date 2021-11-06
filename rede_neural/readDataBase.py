import numpy as np
import os, sys
import imageio

def readPetroBranco():
    trainPath = ".\\preto_branco\\train"
    testPath = ".\\preto_branco\\test"

    categories = os.listdir(testPath)

    #calculate how many images are in the train and test folders
    imageQtd = 0
    for category in categories:
        currPath = os.path.join(trainPath, category)
        images = os.listdir(currPath)
        imageQtd += len(images)
        
        currPath = os.path.join(testPath, category)
        images = os.listdir(currPath)
        imageQtd += len(images)

    print("imgQtds:", imageQtd)

    #get image size
    imagePath = os.path.join(trainPath, category)
    imagePath = os.path.join(imagePath, os.listdir(imagePath)[1])
    im = imageio.imread(imagePath)
    print("im:", im)
    print("type(im):", type(im))
    imShape = list(im.shape)
    print("imShape:", imShape)


    # create array
    arrayShape = [imageQtd]
    arrayShape.extend(imShape)
    db_x = np.zeros(arrayShape, dtype=np.uint8)
    print("db_x shape:", db_x.shape)

    db_y = np.zeros([imageQtd], dtype=np.uint8)
    print("db_y shape:", db_y.shape)

    # read images
    print("reading images")
    currImg = 0
    for category in categories:
        categoryIndex = categories.index(category)

        currPath = os.path.join(trainPath, category)
        images = os.listdir(currPath)
        for image in images:
            imagePath = os.path.join(currPath, image)
            im = imageio.imread(imagePath)
            db_x[currImg] = im
            db_y[currImg] = categoryIndex

            currImg += 1
        print("progress: ", currImg*100/imageQtd, "%")

        currPath = os.path.join(testPath, category)
        images = os.listdir(currPath)
        for image in images:
            imagePath = os.path.join(currPath, image)
            im = imageio.imread(imagePath)
            db_x[currImg] = im
            db_y[currImg] = categoryIndex

            currImg += 1
        print("progress: ", currImg*100/imageQtd, "%")

    print(db_x)
    print(db_x.shape)
    
    #save array
    file = open('pretobranco_x.npy', 'wb')
    np.save(file, db_x)
    
    file = open('pretobranco_y.npy', 'wb')
    np.save(file, db_y)





readPetroBranco()