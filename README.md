# superai-3-Image-NER

## Introduction

The Super AI Engineer 2022: Image-NER competition is a challenge for participants to develop approaches for predicting named entity recognition from scanned documents stock photos. Image processing techniques and Tesseract OCR are used in this notebook to improve the quality of the input images and facilitate the NER task. The competition dataset is not publicly available, and participants must update the Google Drive path to access their own images. This notebook is intended to be a resource and inspiration for other participants in the competition.

## Outcome

- **Ranked 🎖️#1 amongs 79 teams** participating the [Kaggle hackathon](https://www.kaggle.com/competitions/superai-hackathon-online-image-ner/leaderboard) in the score leaderboard.
- **Scored (Edit distance) 0.33569 and 0.32153** in private and public leaderboard

![image](https://user-images.githubusercontent.com/98932144/208120480-35e027bc-f5d6-4ed9-aadd-f869dd52a8b7.png)

## Requirements
- Transformers (Huggingface)
- Datasets (Huggingface)
- simpletransformers
- deskew
- pytesseract

``` python
!pip -q install transformers
!pip -q install datasets
!pip -q install simpletransformers
!pip -q install python-crfsuite
!pip -q install pytesseract
!pip -q install deskew
```

#### Install the tesseract-ocr
``` python
!sudo apt-get install  tesseract-ocr libtesseract-dev tesseract-ocr-tha
```
## Method for image processing 
#### 1) Reduce noise with fastNlmean

 fastNlMeansDenoising() is a function in the OpenCV (Open Source Computer Vision) library that can be used to reduce noise in images. It is based on the Non-Local   Means Denoising algorithm, which uses an approach called "collaborative filtering" to denoise an image. The basic idea behind this approach is to compare the intensities of each pixel with the intensities of its neighbors, and to average them out if the intensities are similar. This can help to smooth out noise and reduce the amount of detail in the image.

``` python
import cv2
dst = cv2.fastNlMeansDenoising(src, h=3, templateWindowSize=7, searchWindowSize=21)
```

 ![image](https://user-images.githubusercontent.com/98932144/208123314-f0ed0938-1520-45ed-8828-d4722cadcc67.png)
 
 #### 2) Color to gray sclae
  Converting an image from color (also known as "RGB" or "true color") to grayscale is a common image processing operation. Grayscale images are simpler and smaller in size than color images, and they can be easier to analyze and process in some cases.

In OpenCV, you can convert an image from color to grayscale using the cvtColor() function. This function takes an image as input and performs a color conversion based on the specified code. To convert an image from color to grayscale, you can use the following code:


``` python
# Load the color image
img = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

![image](https://user-images.githubusercontent.com/98932144/208124635-c44996fd-81d7-4932-992a-f213dc60a87c.png)

#### 3) Deskew image

Deskewing an image is the process of correcting the skew or rotation of an image. Skew occurs when an image is not perfectly aligned with the horizontal or vertical axis, and it can make the image difficult to analyze or process.

from deskew import determine_skew

``` python
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = determine_skew(cvImage)
    return rotateImage(cvImage, angle)

deskew_ = deskew(gray)

```

![image](https://user-images.githubusercontent.com/98932144/208125124-0c027638-10c8-47dc-98f3-1f023c9d790a.png)

#### 4) GaussianBlur and threshold_OTSU

  Gaussian blur is a smoothing filter that can be used to reduce noise and detail in an image. It works by convolving the image with a Gaussian kernel, which is a matrix of weights that defines the blur strength and radius. In OpenCV, you can apply Gaussian blur to an image using the GaussianBlur() function. Here is an example of how to use this function:


``` python
# Apply Gaussian blur with a kernel size of 5
blurred = cv2.GaussianBlur(img, (5, 5), 0)
```
Otsu's thresholding is a method for automatically selecting a threshold value for image segmentation. It works by maximizing the variance between two classes of pixels (e.g., foreground and background) in the image. In OpenCV, you can apply Otsu's thresholding to an image using the threshold() function with the cv2.THRESH_OTSU flag. Here is an example of how to use this function:


``` python
# Apply Otsu's thresholding
threshold, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
```
![image](https://user-images.githubusercontent.com/98932144/208127272-8cd643ec-2535-48ab-b7a3-6d4db04eed17.png)


#### 5) Erode and detect text

To erode text in an image using OpenCV, you can use the cv2.erode() function. This function applies a structuring element to the image, which erodes away the boundaries of the text. Here's an example of how to use cv2.erode() to erode text in an image:


``` python
# Create a kernel for the erosion
kernel = np.ones((5,14), np.uint8)

# Erode the image using the kernel
eroded = cv2.erode(image, kernel, iterations=2)
```

Find contours in the image using cv2.findContours().

``` python
# Find contours in the image
_, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and draw a rectangle around each one
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
```
![image](https://user-images.githubusercontent.com/98932144/208127671-5a5defa2-822e-4add-ba29-98f864da6bbd.png)

#### 5) Convolution sharp kernel 

To sharpen an image using a convolution kernel in OpenCV, you can use the cv2.filter2D() function. This function applies a convolution kernel to the image, which can be used to sharpen the image by enhancing the high-frequency components.

To create a sharpening kernel, you can use a 3x3 kernel with the following values:

``` python
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
```
This kernel will sharpen the image by enhancing the edges and details in the image. Here's an example of how to use cv2.filter2D() to sharpen an image using this kernel:

``` python
# Sharpen the image using the kernel
sharpened = cv2.filter2D(image, -1, kernel)
```

![image](https://user-images.githubusercontent.com/98932144/208129084-53cf4519-80e9-4a4a-a040-5e4c28eb5e67.png)


