## Synopsis
Boosting is a general method for improving the accuracy of any given learning algorithm. Specifically, one can use it to combine weak learners, each performing only
slightly better than random guess, to form an arbitrarily good hypothesis. In this
project, you are required to implement an AdaBoost and RealBoost algorithms for
frontal human face detection.

<b>Part	1: Construct weak classifiers </b><br/>
Construct weak classifiers by loading the predefined set of Haar filters. Compute the features by applying each Haar filter to the integral images of the
positive and negative populations. Determine the polarity and threshold
for each weak classifier. Write a function which returns the weak classifier with lowest weighted error.
Note, as the samples change their weights over time, the histograms and threshold theta will change.

<b>Part	2: Implement Adaboost</b><br/>
Implement the AdaBoost algorithm to boost the weak
classifiers. Construct the strong classifier H(x) as an weighted ensemble of T
weak classifiers. Two class photos are given. Scale the image into a few scales so
that the faces at the front and back are 16x16 pixels in one of the scaled image.
Run your classifier on these images. Perform non-maximum suppression, i.e. when two positive detections overlap significantly, choose the one that has higher
score. Perform hard negatives mining. You are given background images without
faces. Run your strong classifier on these images. Any "faces" detected by your
classifier are called "hard negatives". Add them to the negative population in
the training set and re-train your model. Include the following in your report:
1) Haar filters: Display the top 20 Haar filters after boosting. Report the
corresponding voting weights.
2) Training error of strong classifier: Plot the training error of the strong
classifier over the number of steps T.
3) Training errors of weak classifiers: At steps T = 0; 10; 50; 100, plot the
curve for the training errors of the top 1; 000 weak classifiers among the pool
of weak classifiers in increasing order. Compare these four curves.
4) Histograms: Plot the histograms of the positive and negative populations
over F(x), for T = 10; 50; 100, respectively.
5) ROC: Based on the histograms, plot their corresponding ROC curves for
T = 10; 50; 100, respectively.
6) Detections: Display the detected faces in both of the provided images
without hard negative mining.
7) Hard negative mining: Display the detected faces in both of the provided
images with hard negative mining.

<b>Part 3: Implement Realboost</b><br/>
Implement RealBoost: Implement the RealBoost algorithm using the top
T = 10; 50; 100 features chosen in step 3). Compute the histograms of neg-
ative and positive populations and the corresponding ROC curves. Include the
following in your report:
8) Histograms: Plot the histograms of the positive and negative populations
over F(x), for T = 10; 50; 100, respectively.
9) ROC: Based on the histograms, plot their corresponding ROC curves. Compare them with the ROC curves in 5).

## Motivation

We can compare the performance of adaboosting and realboosting in this project.

## Acknowledgements

This machine learning project is part of UCLA's Statistics 231/Computer Science 276A course on pattern recognition in machine learning, instructed by Professor Song-Chun Zhu.
