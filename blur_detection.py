import cv2
import numpy as np



img = cv2.imread("0.jpg", cv2.IMREAD_GRAYSCALE)

print("IMG: 0.jpg")

med_val = np.median(img) 
lowerEdge = int(max(0, 0.7 * med_val))
upperEdge = int(min(255, 1.3 * med_val))

print("edges_var =", cv2.Canny(img, lowerEdge, upperEdge).var())
print("laplacian_var =", cv2.Laplacian(img, cv2.CV_64F).var())
print("sobelx_var =", cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).var())
print("sobely_var =", cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).var())


img = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

print("IMG: 1.jpg")

med_val = np.median(img) 
lowerEdge = int(max(0, 0.7 * med_val))
upperEdge = int(min(255, 1.3 * med_val))

print("edges_var =", cv2.Canny(img, lowerEdge, upperEdge).var())
print("laplacian_var =", cv2.Laplacian(img, cv2.CV_64F).var())
print("sobelx_var =", cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).var())
print("sobely_var =", cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).var())


img = cv2.imread("2.jpg", cv2.IMREAD_GRAYSCALE)

print("IMG: 2.jpg")

med_val = np.median(img) 
lowerEdge = int(max(0, 0.7 * med_val))
upperEdge = int(min(255, 1.3 * med_val))

print("edges_var =", cv2.Canny(img, lowerEdge, upperEdge).var())
print("laplacian_var =", cv2.Laplacian(img, cv2.CV_64F).var())
print("sobelx_var =", cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5).var())
print("sobely_var =", cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5).var())


exit()

cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()