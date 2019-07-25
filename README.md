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
- [Bivariate function approximation with linear regression](linear/LinearRegressionBivariate.html)

## Classification

Classification is the other side of the coin in Data Science.

### Binary classification

Let's start with the binary classification and logistic regression and add some more refinements:
- [Univariate function as boundary on a two classes data, approximated with logistic regression](classification/ClassificationContinuousSingleFeature.html)
- [Bivariate parametric function as a boundary, approximated with logistic regression](classification/ClassificationContinuous2Features.html)
  - [Using Tensor flow](classification/ClassificationContinuous2Features-TensorFlow.html)
  - [Using Keras](classification/ClassificationContinuous2Features-Keras.html), adding regularizers and eventually a two layer neural net
  
  
# Reading list

## Nice notebooks

- [Probabilistic programming and Bayesian methods for hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)