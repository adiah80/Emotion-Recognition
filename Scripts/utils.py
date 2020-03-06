from lib import *

def pad_image(image, max_width, detach=True):
    if(detach): img = image.detach().numpy()
    cur = image.shape[2]
    req = max_width - cur
    return np.pad(img,((0,0),(0,0),(0,req)), 'constant')