import numpy as np
from matplotlib import pyplot as plt 
from PIL import Image
# from PIL import ImageDraw
# feat_points is a list of integers

def findBox(feat_points):
    # conver to points
    feat_array = np.array(feat_points).reshape((-1,2))
    p_left  = feat_array[:,0].min()
    p_right = feat_array[:,0].max()
    p_up    = feat_array[:,1].min()
    p_down  = feat_array[:,1].max()
    return p_left,p_up,p_right,p_down
    
# type of im is PIL.Image
def preprocess(im, box, out_size=(200,200),edge=False):
#     draw = ImageDraw.Draw(im_new)
#     draw.rectangle((p_left,p_up,p_right,p_down))


    im_new = img.crop(box) # sub-region of image
    im_new = im_new.resize(out_size) # square image
    im_new = im_new.convert('L') # grayscale
    
    if edge: im_new = im_new.filter(ImageFilter.CONTOUR)
    
    return im_new


# demo
featP =[195,293,269,115,868,158,888,170,641,496,512,546]
box = findBox(featP)
img_new = preprocess(img,box,out_size=(100,100))