'''
EDGE DETECTION 
Edge detection is a fundamental image processing technique used to identify the boundaries or edges of objects in an image.
it is a multi stage algorithm that involves the following steps:

1. noise reduction: remove the noise in the image with a 5x5 Gaussian filter.
2. finding the intensity gradients of the image: Smoothened image is then filtered with a Sobel kernel in both horizontal and vertical direction
3. non maximum suppression: remove any unwanted pixels which may not constitute the edge.
4. Double Thresholding: Classify edges as strong or weak using high and low threshold values.
'''

'''
# 1. Noise reduction -> Gaussian filter
import cv2
import numpy as np
image = cv2.imread('image_with_alpha.png',0) # we want to load it as a grayscale image
# applying a 5x5 Gaussian filter
blurred_image = cv2.GaussianBlur(image,(5,5),1.4)
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Blurred Image', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2. Finding the intensity gradients -> Sobel filter
# applying a Sobel filter in both horizontal and vertical direction
# Gradients represent changes in intensity in an image, and 
# the Sobel operator is used to compute the gradients in both x and y directions
# edges apply where the gradient is high, and non-edges apply where the gradient is low
sobel_x = cv2.Sobel(blurred_image,cv2.CV_64F,1,0,ksize=3)
sobel_y = cv2.Sobel(blurred_image,cv2.CV_64F,0,1,ksize=3)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Non-maximum suppression
# Non-Maximum Suppression (NMS) is applied to thin the edges. 
# It works by preserving only the pixels that have the highest gradient value in their neighborhood, 
# along the direction of the gradient.
# by direction of gradient we mean the direction of the maximum gradient in the image
def non_maximum_suppression(gradient_magnitude, gradient_direction):
    M, N = gradient_magnitude.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = gradient_direction * 180.0 / np.pi  # Convert to degrees
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                # Angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                # Angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                # Angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    Z[i, j] = gradient_magnitude[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z

gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_direction = np.arctan2(sobel_y, sobel_x) * (180 / np.pi)  # Convert radians to degrees

image = non_maximum_suppression(gradient_magnitude, gradient_direction)

cv2.imshow('Non-Maximum Suppression', image / np.max(image))  # Normalize for viewing
cv2.waitKey(0)

# double thresholding 
high_threshold = 75
low_threshold = 25
strong = 255
weak = 75
# Create output image
thresholded_image = np.zeros_like(image)
strong_i, strong_j = np.where(image >= high_threshold)
weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))
thresholded_image[strong_i, strong_j] = strong
thresholded_image[weak_i, weak_j] = weak
cv2.imshow('Double Threshold', thresholded_image)
cv2.waitKey(0)
# connecting the non isolated weak edges
def edge_tracking_by_hysteresis(thresholded_image, weak=75, strong=255):
    M, N = thresholded_image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if thresholded_image[i, j] == weak:
                # Check if any of the surrounding pixels are strong
                if ((thresholded_image[i+1, j-1] == strong) or (thresholded_image[i+1, j] == strong) or (thresholded_image[i+1, j+1] == strong)
                    or (thresholded_image[i, j-1] == strong) or (thresholded_image[i, j+1] == strong)
                    or (thresholded_image[i-1, j-1] == strong) or (thresholded_image[i-1, j] == strong) or (thresholded_image[i-1, j+1] == strong)):
                    thresholded_image[i, j] = strong
                else:
                    thresholded_image[i, j] = 0
    return thresholded_image

final_edges = edge_tracking_by_hysteresis(thresholded_image)
cv2.imshow('Final Edges', final_edges)
cv2.waitKey(0)

canny_edges = cv2.Canny(image, low_threshold, high_threshold)
cv2.imshow('Canny Edges', canny_edges)
cv2.waitKey(0)
'''


import cv2
import numpy as np

def gaussian_blur(image, kernel_size=5):
    """Apply Gaussian Blur to smooth the image and reduce noise."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def sobel_operator(image):
    """Apply Sobel operator to compute gradients along x and y."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)  # Edge strength
    gradient_direction = np.arctan2(sobel_y, sobel_x)      # Edge direction
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """Perform Non-Maximum Suppression to thin the edges."""
    M, N = gradient_magnitude.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = gradient_direction * 180.0 / np.pi  # Convert to degrees
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q, r = 255, 255
                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                # Angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                # Angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                # Angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                    Z[i, j] = gradient_magnitude[i, j]
                else:
                    Z[i, j] = 0
            except IndexError:
                pass

    return Z

def double_threshold(image, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    """Apply double thresholding to classify strong and weak edges."""
    high_threshold = image.max() * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio
    strong = 255
    weak = 75

    strong_edges = (image >= high_threshold).astype(np.uint8) * strong
    weak_edges = ((image >= low_threshold) & (image < high_threshold)).astype(np.uint8) * weak

    return strong_edges + weak_edges

def edge_detection_pipeline(image_path):
    """Complete Edge Detection Pipeline."""
    image = cv2.imread('image_with_alpha.png', cv2.IMREAD_GRAYSCALE)
    smoothed_image = gaussian_blur(image)
    gradient_magnitude, gradient_direction = sobel_operator(smoothed_image)
    thin_edges = non_maximum_suppression(gradient_magnitude, gradient_direction)
    final_edges = double_threshold(thin_edges)
    cv2.imshow('Edge Detection Result', final_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

edge_detection_pipeline('image.jpg')
