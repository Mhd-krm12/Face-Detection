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


#Face detection model files (using "OpenCV" and "Caffe pretrained model")
protoPath = "deploy.prototxt.txt"
modelPath = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


#image to matrix
for file in get_image_files(r"C:\Users\Mhd Krm\Desktop\face_de\deneme"):
    count = 0
    print(file)
    resim = cv2.imread(file)
    try:
        resim_rgb = resim[...,::-1]          # BGR to RGB
        (h, w) = resim_rgb.shape[:2]         # Get the height and width of the image 
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(resim_rgb, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)         # give image to model 
        detections = detector.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            
            #benzerlik similarity rate 
            if confidence > 0.5:
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                #Draw a square around the face
                resim2=cv2.rectangle(resim, (startX, startY), (endX, endY), (0, 255, 0),1) 

                #find the face and crop
                only_face = resim[startY:endY, startX:endX]    
                only_face = only_face[:, :, [2, 1, 0]]         #RGB to BGR

                #image show 
                plt.imshow(only_face)
                plt.show(block=False)
                plt.pause(1)
                plt.close() 

                #image labeling: create text file and write face coordinate
                file_name=os.path.split(file)[-1].split('.')[-2]
                selecteed_folder=(f"C:/Users/Mhd Krm/Desktop/face_de/deneme/{file_name}")  
                fi= open(selecteed_folder+'.txt' , "a")
                bx = ((endX - ((endX - startX)) / 2) / w )
                by = ((endY - ((endY - startY)) / 2) / h ) 
                bw = ((endX - startX) / w) 
                bh = ((endY- startY) / h )
                fi.write("1"+ " " +str(bx) + " " + str(by)  + " " + str(bw) + " " + str(bh) + "\n")

        print(" The number of faces: " + str(count))
  
        if count == 0:
            print(str(file))
            Pictures_without_faces_counter+=1
            print(" This image does not contain faces, it will be deleted. ")
            os.remove(file)

    except:
        invalid_image_counter=+1
        print(" This image is invalid, it will be deleted. ")
        os.remove(file)

     
print("\n")                
print(str(Pictures_without_faces_counter)+" image do not contain faces, is deleted.")
print(str(invalid_image_counter)+" image are invalid, is deleted.")
