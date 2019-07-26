# Learning data science

Most of the examples presented in tutorials are either using powerful libraries, or based on data samples with many features.

In this collection of workbooks, we want to start from simple examples and raw Python code and then progressively complexify the data sets and use more complex technics, libraries and datasets.

On purpose, used data is generated as it allows for more flexibility as parameters are modifiable.

The notebooks are of type Jupyter, using Python 3.

![Do not get confused](assets/Confused_640.png)

## Linear regression

Linear regression is the mother of Data Science algorithm.

Let's progressively start from simple univariate example and then add progressively more complexity:
- [Univariate function approximation with linear regression](/linear/LinearRegressionUnivariate.html), closed form, with Numpy, Scipy or SKLearn, eventually with gradient descent and stochastic gradient descent
  - [Using Tensor Flow](linear/LinearRegressionUnivariate-TensorFlow.html)
- [Bivariate function approximation with linear regression](linear/LinearRegressionBivariate.html), closed formed, with SKLearn, (stochastic) gradient descent, regularizer

## Classification

Classification is the other side of the coin in Data Science.

### Binary classification with parametric model

Let's start with the binary classification and logistic regression and add some more refinements:
- [Univariate function as boundary on a two classes data, approximated with logistic regression](classification/ClassificationContinuousSingleFeature.html)
- [Bivariate parametric function as a boundary, approximated with logistic regression](classification/ClassificationContinuous2Features.html)
  - [Using Tensor flow](classification/ClassificationContinuous2Features-TensorFlow.html)
  - [Using Keras](classification/ClassificationContinuous2Features-Keras.html), adding regularizers and eventually a two layer neural net
  
### Binary classification with non-parametric model

- [Bivariate with K Nearest Neighbors (KNN)](classification/ClassificationContinuous2Features-KNN.html)
  
# Reading list

## Nice notebooks

- [Probabilistic programming and Bayesian methods for hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

## Papers

- [You Look Only Once: Unified, Real-time object detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
  - [YOLO in Keras - Machine Learning Mastery](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)