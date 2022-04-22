import cv2, os
from matplotlib import pyplot as plt
import numpy as np
import glob

Pictures_without_faces_counter=0
invalid_image_counter=0

#Get all Image path in folder.
def get_image_files(folder_path):
    img_formats = ['jpg','png', 'jpeg'] 
    files = []
    for image_file in glob.iglob(os.path.join(folder_path, "**"), recursive=True):
        files.append(image_file)
    return [x for x in files if x.split('.')[-1].lower() in img_formats]
