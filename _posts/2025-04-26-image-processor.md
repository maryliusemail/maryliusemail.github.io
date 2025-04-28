---
title: Image Proccesor + Classification Program
date: 2025-04-26 21:05:32 +/-TTTT
categories: [data science, python]
tags: [projects]     # TAG names should always be lowercase
description: This project builds a basic image processing system and a simple machine learning classifier from scratch.
image:
  path: /assets/img/previews/image-processor-preview.png
  # alt: image alternative text
---
## 🚀 How to Run the Project
[](https://github.com/maryliusemail/Image-Processing-and-Classification-Program)

## ✨ Core Modules and Functionalities

### RGBImage Class
Defines the fundamental RGB image object using nested lists:
- Initializes 3D matrices representing pixels in `(row, column, [R,G,B])` format.
- Enforces strict type and shape validation with runtime exception handling.
- Supports controlled access to image data through getters/setters.
- Implements deep copying to maintain image immutability across transformations.


---

### ImageProcessingTemplate Class
Provides a library of stateless, computationally efficient image transformations:

- **Negate**: Invert pixel intensities (`255 - intensity`) across all color channels.

_Example:_  
![ezgif com-animated-gif-maker (6)](/assets/img/proj_gif/image-processor/ezgif.com-animated-gif-maker_6.gif)



- **Grayscale**: Average RGB channels per pixel using floor division.

_Example:_  
![ezgif com-crop](/assets/img/proj_gif/image-processor/ezgif.com-crop.gif)


- **Rotate 180°**: Flip the image along both horizontal and vertical axes.

_Example:_  
![ezgif com-animated-gif-maker (5)](/assets/img/proj_gif/image-processor/ezgif.com-animated-gif-maker_5.gif)



- **Adjust Brightness**: Add/subtract uniform intensity with clipping at `[0, 255]`.

_Example:_  
![ezgif com-resize](/assets/img/proj_gif/image-processor/ezgif.com-resize.gif)


- **Blur**: Smooth image using local neighborhood averaging.

_Example:_  
![ezgif com-animated-gif-maker (3)](/assets/img/proj_gif/image-processor/ezgif.com-animated-gif-maker_3.gif)



---

### StandardImageProcessing Class
Extends `ImageProcessingTemplate` with minor usage tracking:
- Inherits all transformation operations.
- Tracks the number of processing operations performed.
- Provides a coupon system to allow free operations temporarily.


---

### PremiumImageProcessing Class
Extends the base functionality with additional advanced image manipulation operations:

- **Tile**: Repeats an image to fill larger dimensions using modular indexing.

_Example:_  
![ezgif com-animated-gif-maker (7)](/assets/img/proj_gif/image-processor/ezgif.com-animated-gif-maker_7.gif)


- **Sticker**: Overlays a smaller image onto a background at a specified (x, y) coordinate.

_Example:_  
![ezgif com-animated-gif-maker (1)](/assets/img/proj_gif/image-processor/ezgif.com-animated-gif-maker_1.gif)


- **Edge Highlight**: Applies a 3x3 Laplacian convolution filter to detect and highlight edges.

_Example:_  
![ezgif com-animated-gif-maker](/assets/img/proj_gif/image-processor/ezgif.com-animated-gif-maker.gif)

---

## 🧠 Image KNN Classifier Overview

K-Nearest Neighbors (KNN) is a classic machine learning algorithm commonly used for **classification** tasks.  
It works under the principle that similar data points exist close together in feature space.

In this project, we apply KNN to classify images based on their raw pixel values.

---

### 🌞🌙 Real-World Example: Classifying Day vs Night Images

This project follows a typical KNN workflow.  
Imagine building a model to determine whether an image shows **daytime** or **nighttime**:

1. **Collect a dataset**:
   - Images labeled `"daytime"`
   - Images labeled `"nighttime"`

2. **Classify a new image**:
   - Measure how **similar** the new image is to the labeled examples.
   - Find the **k** closest images (nearest neighbors).
   - **Vote** among them to predict the label.

This approach generalizes to **any kind of image classification** based on visual similarity.


## 🛠️ How the KNN Classifier Works in This Project

### **Step 1: Fitting the Model — `fit(data)`**

- **Purpose**: Save labeled training data (images and labels) for future use.
- **Input**: List of `(image, label)` pairs.


---

### **Step 2: Measuring Distance — `distance(image1, image2)`**

- **Purpose**: Quantify how visually similar two images are.
- **Method**:
  - Flatten each 3D RGB image matrix into a 1D list.
  - Compute the **Euclidean distance** between corresponding pixel intensities:
  
    $$d(a, b) = \sqrt{(a_1-b_1)^2 + (a_2-b_2)^2 + \dots + (a_n-b_n)^2}$$

  
  - A **smaller distance** indicates higher similarity.

_Image 1:_  
![steve](/assets/img/proj_gif/image-processor/steve.png)

_Image 2:_  
![knn_test_img](/assets/img/proj_gif/image-processor/knn_test_img.png)


```python
img1 = img_read_helper('img/steve.png')
img2 = img_read_helper('img/knn_test_img.png')
knn = ImageKNNClassifier(3)
knn.distance(img1, img2)
```
- Output: 15946.312896716909
---

### **Step 3: Voting — `vote(candidates)`**

- **Purpose**: Choose the most common label among the nearest neighbors.
- **Input**: A list of candidate labels (strings).
- **Behavior**:
  - Returns the most frequent label.
  - If there’s a tie, any of the majority labels may be selected.


---

### **Step 4: Predicting — `predict(image)`**

- **Purpose**: Predict the label of a new image using the KNN method.
- **Workflow**:
  1. Compute distances to all stored training images.
  2. Sort the images by ascending distance.
  3. Select the top `k_neighbors`.
  4. Apply `vote()` to predict the label based on the neighbors' labels.

- **Training Data**:
  - Training images are loaded from the `knn_data/` folder.
  - Each image is labeled (e.g., `"daytime"`, `"nighttime"`) and used for nearest neighbor comparisons.


---



_Example:_  
![knn_test_img](/assets/img/proj_gif/image-processor/knn_test_img.png)

```python
knn_tests('img/knn_test_img.png')
```
- Output: nighttime

- ✅ **Result**: The model correctly predicted the image as nighttime!



---


