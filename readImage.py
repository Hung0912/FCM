import PIL
from PIL import Image
import numpy as np
from matplotlib import image
from os import listdir

#Use Pillow
# print("Pillow version:", PIL.__version__)


def loadImg(imageName):
    image = Image.open(imageName)
    print('Format:',image.format, ', Mode:', image.mode,', Size:', image.size)
    data = np.asarray(image)
    image2 = Image.fromarray(data)
    image2.show()
    return data


#Use matplotlib
def loadImage(imageName):
    data = image.imread(imageName)
    print(data)
    print(data.dtype)
    print(data.shape)
    return data

#load all images in direc
def loadImageFromFile(directory):
    loaded_images = list()
    for filename in listdir(directory):
    	# load image
    	img_data = image.imread(directory + '/'+ filename)
    	# store loaded image
    	loaded_images.append(img_data)
    	print('> loaded %s %s' % (filename, img_data.shape))

def convertImageToGrayscale(image_name):
    image = Image.open(image_name)
    # convert the image to grayscale
    gs_image = image.convert(mode='L')
    # save in jpeg format
    gs_image.save('test_grayscale.jpg')
    # load the image again and show it
    image2 = Image.open('test_grayscale.jpg')
    # show the image
    image2.show()
    data = np.asarray(image2)
    print(data.shape)


if __name__ == "__main__":
    # loadImageFromFile('images')
    # convertImageToGrayscale('images/test.jpg')
    data = loadImg('images/test.jpg')
    print(data.shape)
    print(data[0,0])