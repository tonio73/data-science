# Learning data science step by step

Most of the examples presented in Internet tutorials are either using powerful libraries (Scikit Learn, Keras...), complex models (neural nets), or based on data samples with many features.

In this collection of workbooks, I want to start from simple examples and raw Python code and then progressively complexify the data sets and use more complex technics and libraries.

On purpose, datasets are generated in order to adjust the parameters fitting with the demonstration.

The notebooks are of type Jupyter, using Python 3.

__Best view on the notebooks are :__
- Browser notebooks in HTML from [the HTML table of content](https://tonio73.github.io/data-science/)
- Open this repository [in nbviewer](https://nbviewer.jupyter.org/github/tonio73/data-science/tree/master/)
- Clone the repository in order to test and modify locally within Jupyter ou JupyterLab

![Do not get confused](assets/Confused_640.png)

## Linear regression

Linear regression is the mother of Data Science algorithm.

Let's progressively start from simple univariate example and then add progressively more complexity:
- __Univariate function approximation with linear regression__, closed form, with Numpy, Scipy or SciKit Learn, eventually with gradient descent and stochastic gradient descent ([HTML](linear/LinearRegressionUnivariate.html) / [Notebook](linear/LinearRegressionUnivariate.ipynb))
  - Using Tensor Flow ([HTML](linear/LinearRegressionUnivariate-TensorFlow.html) / [Jupyter](linear/LinearRegressionUnivariate-TensorFlow.ipynb))
- __Bivariate function approximation with linear regression__, closed formed, using SciKit Learn, (stochastic) gradient descent, adding regularizer ([HTML](linear/LinearRegressionBivariate.html) / [Jupyter](linear/LinearRegressionBivariate.ipynb))
  - Using Keras, single perceptron linear regression, two layer model ([HTML](linear/LinearRegressionBivariate-Keras.html) / [Jupyter](linear/LinearRegressionBivariate-Keras.ipynb)) 
  - Model confidence and quality evaluation in case of the Gaussian model ([HTML](linear/LinearRegressionBivariateQuality.html) / [Jupyter](linear/LinearRegressionBivariateQuality.ipynb))
- Feature engineering or feature learning with linear regression ([HTML](linear/LinearRegressionFeatureEngineering-Keras.html) / [Jupyter](linear/LinearRegressionFeatureEngineering-Keras.ipynb))

## Classification

Classification is the other side of the coin in Data Science.

### Binary classification with parametric models

Let's start with the binary classification and logistic regression and add some more refinements:
- __Univariate function__ as boundary on a two classes data, __approximated with logistic regression__, homemade, using SciKit Learn ([HTML](classification/ClassificationContinuousSingleFeature.html) / [Jupyter](classification/ClassificationContinuousSingleFeature.ipynb))
- __Bivariate parametric function__ as a boundary, __approximated with logistic regression__, homemade, using SciKit Learn ([HTML](classification/ClassificationContinuous2Features.html) / [Jupyter](classification/ClassificationContinuous2Features.ipynb))
  - Using Tensor flow ([HTML](classification/ClassificationContinuous2Features-TensorFlow.html) / [Jupyter](classification/ClassificationContinuous2Features-TensorFlow.ipynb))
  - Using Keras, adding regularizers and eventually a two layer neural net ([HTML](classification/ClassificationContinuous2Features-Keras.html) / [Jupyter](classification/ClassificationContinuous2Features-Keras.ipynb))
  
### Binary classification with non-parametric models

Beyond (linear) regression, non-parametric models:
- __Bivariate with K Nearest Neighbors (KNN)__, homemade, using SciKit Learn ([HTML](classification/ClassificationContinuous2Features-KNN.html) / [Jupyter](classification/ClassificationContinuous2Features-KNN.ipynb))

### Multi-class regression

Going further with more than two classes or categories:
- __Two features to separate the 2D plan into 3 or more categories__
  - Using Keras matching on __linearly separable problem__ (Czech flag) and __not linearly separable problem__ (Norway flag), using 2 and 3 layer neural net to handle the second problem ([HTML](classification/ClassificationMulti2Features-Keras.html) / [Jupyter](classification/ClassificationMulti2Features-Keras.ipynb))
  
 ### Multi-class classification with non-parametric models 

- __Multi-class classification using decision trees__ ([HTML](classification/ClassificationMulti2Features-Tree.html) / [Jupyter](classification/ClassificationMulti2Features-Tree.ipynb))

# Reading list

## Books

- Deep Learning - I. Goodfellow, Y. Bengio, A. Courville, The MIT Press. 
    - Very good overview of machine learning and its extension to deep learning
- An Introduction to Statistical Learning with Applications in R - G. James, D. Witten, T. Hastie, R. Tibshirani. 
    - Traditional machine learning including regressions, clustering, SVM...

## Nice notebooks

- [Probabilistic programming and Bayesian methods for hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)

## Tutorials and courses

- [Deep learning tutorial - Stanford](http://ufldl.stanford.edu/tutorial/)
- [High dimension statistics - MIT OpenCourseware 18s997, 2015](https://ocw.mit.edu/courses/mathematics/18-s997-high-dimensional-statistics-spring-2015/lecture-notes/)

## Papers

- [You Look Only Once: Unified, Real-time object detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)
  - [YOLO in Keras - Machine Learning Mastery](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)
- [Learning to forget, continual prediction with LSTM - F. A. Gers et al.](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.55.5709&rep=rep1&type=pdf)
- [What are biases in my word embeddings ? - N. Swinger et al.](https://arxiv.org/pdf/1812.08769.pdf)
  
  
## Articles

- Build the right autoencoder — Tune and Optimize using PCA principles - Medium [Part I](https://medium.com/@cran2367/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-i-1f01f821999b), [Part II](https://medium.com/@cran2367/build-the-right-autoencoder-tune-and-optimize-using-pca-principles-part-ii-24b9cca69bd6)

## Data / model sources

### Word embeddings & analysis

- [ConceptNet](http://www.conceptnet.io/)
- [GloVe: Global Vectors for Word Representation - Stanford](https://nlp.stanford.edu/projects/glove/)
- [Opinion Mining, Sentiment Analysis, and Opinion Spam Detection](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)
