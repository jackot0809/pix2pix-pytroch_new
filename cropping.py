import cv2
import os
from tqdm import tqdm

for filename in tqdm(os.listdir('/home/jack/tasks/pix2pix-pytorch/enhancedlip/test')):
    f = os.path.join('/home/jack/tasks/pix2pix-pytorch/enhancedlip/test',filename)
    img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    resized = img[0:256,0:256]
    #resized = cv2.GaussianBlur(img,(15,15),0)
    cv2.imwrite(f,resized)