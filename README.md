# Notebooks on learning data science

Most of the examples presented in tutorials are either using powerful libraries, or data samples with many features.

In this collection of workbooks, we want to start from simple examples and raw Python code and then progressively complexify the data sets and use more complex technics and libraries.

On purpose, these work books are based on generated data more than on experimentaly collected data. Generated data allows for more flexibility as parameters are modifiable.

## Linear regression

Linear regression is the mother of Data Science algorithm.

Let's progressively start from simple univariate example and then add progressively more complexity:
- [Univariate function approximation with linear regression](/linear/LinearRegressionUnivariate.ipynb)
  - [Using Tensor Flow](linear/LinearRegressionUnivariate-TensorFlow.ipynb)
- [Bivariate function approximation with linear regression](linear/LinearRegressionBivariate.ipynb)

## Classification

Classification is the other side of the coin in Data Science.

Let's start with the binary classification and logistic regression and add some more refinements:
- [Univariate function as boundary on a two classes data, approximated with logistic regression](classification/ClassificationContinuousSingleFeature.ipynb)
- [Bivariate parametric function as a boundary, approximated with logistic regression](classification/ClassificationContinuous2Features.ipynb)
  - [Using Tensor flow](classification/ClassificationContinuous2Features-TensorFlow.ipynb)
  - [Using Keras](classification/ClassificationContinuous2Features-Keras.ipynb), adding regularizers and eventually a two layer neural net