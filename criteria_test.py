import cv2
import numpy as np
import os

images = os.listdir('./test_frames')
num_images = len(images)

data = [] # using an array is more convenient for tabulate.

for i in range(num_images):
    img = cv2.imread('./test_frames/' + images[i])
    
    med_val = np.median(img) 
    lowerEdge = int(max(0, 0.7 * med_val))
    upperEdge = int(min(255, 1.3 * med_val))

    edges_var = cv2.Canny(img, lowerEdge, upperEdge).var()
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    sobelx_var = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).var()
    sobely_var = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).var()
        
    avgR = np.mean(img[:,:,2])
    avgG = np.mean(img[:,:,1])
    avgB = np.mean(img[:,:,0])
    
    print([images[i], str("{:.2f}".format(edges_var)), str("{:.2f}".format(laplacian_var)), int(sobelx_var), int(sobely_var), int(avgR), int(avgG), int(avgB)])