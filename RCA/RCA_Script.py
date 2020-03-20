import RCA
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = 'C:/Users/Danny/PycharmProjects/RCA/test.jpg'

RCA1 = RCA.RetinalCompression()
im = RCA1.load_image(image)
RCA1.show_image(im)
im2, msk = RCA1.distort_image(image=im)
im3 = RCA1.mask(im2,msk)
RCA1.show_image(im3)