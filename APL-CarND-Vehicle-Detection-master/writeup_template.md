##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/hog.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_window.jpg
[image4]: ./examples/heat_map.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/car-notcar.png
[video1]: ./tracked2.mp4

Vehicle tracking is an important step for a self driving car, the state of the art is using deep neural network to achive it but since this algorithm works almost as a black box it is important to take other insights that provides a better picture of how it works. In order to do that here is used a more traditional technique for image recognition, extracting the features by hand (instead of a convolutiona neural network that learns those features) of the image with a descriptor like the histogram of oriented gradients and a classifier like the support vector machine would give a better understanding of how the problem is solved since is needed that the programmer choose the features that would be most usefull. Once the model is created, is applied to a video where in everyframe is expected to detect the cars on the road.


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

For this project is used a dataset of 8792 images of cars and 8968 images of the content of an image that are not cars, for example the nature in the background and lines on the road. Is important to provide the classifier the target images and the not target images. Each sample has 64 x 64 pixels represented in RGB color space. 

![alt text][image8]

The features as it was pointed before needed to be choosen and extracted by the programmer, the most used are Haar filters and the histogram of oriented gradient. For this exercise is HOG the one applied since the model can be created without so much time training. HOG features shows the orientation of the gradient in the cells that are part of an image, the programmer can costumize the quantity of cells that considers important for the problem and then in a histogram is represented where the strongest changes of the gradient in that block. Since the size of the image is 64 x 64 the choosen size of the cell is 8 x 8, where each block contains 8 x 8 pixe. The orientation of the gradient can be represented in 9 bins and then finally been stored in a feature vector of 1 dimension. 

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and campared the following outputs: the accuracy, quality of bounding boxes, false positives, false negatives  and HOG visualization; all of these in the 6 sample images, then the configuration was applied to the whole classification system.



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG and color features as inputs, used format features with np.vstack and StandardScaler(), the data was shuffled and splited into training and testing set, the SVM was training using sklearn.svm.LinearSVC().



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The search for windows was executed with the function slide_window and the space search was limited to the lower part of the image (since in the only part where the cars were expected and would make our computation time and resources more efficient). For every single window it was performed a feature extraction, scalation of the extracted features that go to the classifier, perform the classification by the SVM and store the window if a positive prediction was made.



####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The feature extraction is one of the biggest obstacles, since it needs to tweek some parameter in order to work the best in this particular problem (is not reliable to use this confifuration in other problems). It was needed a good amount of time to create the best configuration possible. In the other hand looking at another aproach besides neural networks has shown how the features were extracted before the CNN beacme the state of the art for image recognition, certainly this insight shows a better understanding of how to solve the problem insted of using deep learning almost like a black box.

