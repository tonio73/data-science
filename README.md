# Learning data science step by step

Most of the examples presented in Internet tutorials are either using powerful libraries (Scikit Learn, Keras...), complex models (neural nets), or based on data samples with many features.

In this collection of workbooks, I want to start from simple examples and raw Python code and then progressively complexify the data sets and use more complex technics and libraries.

On purpose, most datasets are generated in order to adjust the parameters fitting with the demonstration.

The notebooks are of type Jupyter, using Python 3.7

__To read or edit the notebooks you may :__
- Browse notebooks in HTML from [the HTML table of content](https://tonio73.github.io/data-science/)
- Open this repository [in nbviewer](https://nbviewer.jupyter.org/github/tonio73/data-science/tree/master/)
- Clone the repository in order to test and modify locally within Jupyter ou JupyterLab

![Do not get confused](assets/Confused_640.png)

## Linear regression

Let's progressively start from simple univariate example and then add progressively more complexity:
- __Univariate function approximation with linear regression__, 
  - Closed form, with Numpy, Scipy or SciKit Learn, eventually with gradient descent and stochastic gradient descent ([HTML](linear/LinearRegressionUnivariate.html) / [Notebook](linear/LinearRegressionUnivariate.ipynb))
  - Using Tensor Flow ([HTML](linear/LinearRegressionUnivariate-TensorFlow.html) / [Jupyter](linear/LinearRegressionUnivariate-TensorFlow.ipynb))
- __Bivariate function approximation with linear regression__, 
  - Closed form, using SciKit Learn, (stochastic) gradient descent, adding regularizer ([HTML](linear/LinearRegressionBivariate.html) / [Jupyter](linear/LinearRegressionBivariate.ipynb))
  - Using Keras, single perceptron linear regression, two layer model ([HTML](linear/LinearRegressionBivariate-Keras.html) / [Jupyter](linear/LinearRegressionBivariate-Keras.ipynb)) 
  - Model confidence and quality evaluation in the Gaussian model case ([HTML](linear/LinearRegressionBivariateQuality.html) / [Jupyter](linear/LinearRegressionBivariateQuality.ipynb))
- __Feature engineering or feature learning with linear regression__ ([HTML](linear/LinearRegressionFeatureEngineering-Keras.html) / [Jupyter](linear/LinearRegressionFeatureEngineering-Keras.ipynb))

## Classification

### Binary classification with parametric models

- __Univariate function__ as boundary on a two classes data, __approximated with logistic regression__, 
  - Homemade, using SciKit Learn ([HTML](classification/ClassificationContinuousSingleFeature.html) / [Jupyter](classification/ClassificationContinuousSingleFeature.ipynb))
- __Bivariate parametric function__ as a boundary, __approximated with logistic regression__, 
  - Homemade, using SciKit Learn ([HTML](classification/ClassificationContinuous2Features.html) / [Jupyter](classification/ClassificationContinuous2Features.ipynb))
  - Using Tensor flow ([HTML](classification/ClassificationContinuous2Features-TensorFlow.html) / [Jupyter](classification/ClassificationContinuous2Features-TensorFlow.ipynb))
  - Using Keras, adding regularizers and eventually a two layer neural net ([HTML](classification/ClassificationContinuous2Features-Keras.html) / [Jupyter](classification/ClassificationContinuous2Features-Keras.ipynb))
  
### Binary classification with non-parametric models

- __Bivariate with K Nearest Neighbors (KNN)__, homemade, using SciKit Learn ([HTML](classification/ClassificationContinuous2Features-KNN.html) / [Jupyter](classification/ClassificationContinuous2Features-KNN.ipynb))
- Non linear problem solving with __Support Vector Machine (SVM)__ ([HTML](classification/ClassificationSVM.html) / [Jupyter](classification/ClassificationSVM.ipynb))

### Multi-class classification with regression or neural networks

- __Two features to separate the 2D plan into 3 or more categories__
  - Using Keras matching on __linearly separable problem__ (Czech flag) and __not linearly separable problem__ (Norway flag), using 2 and 3 layer neural net to handle the second problem ([HTML](classification/ClassificationMulti2Features-Keras.html) / [Jupyter](classification/ClassificationMulti2Features-Keras.ipynb))
  
### Multi-class classification with non-parametric models 

- __Multi-class classification using decision trees__ ([HTML](classification/ClassificationMulti2Features-Tree.html) / [Jupyter](classification/ClassificationMulti2Features-Tree.ipynb))

### Convolutional neural networks (CNN)

- __Introduction to CNN as an image filter__
    - Part 1 - Horizontal edge detector using a simple 1-2 layer neural nets ([HTML](cnn/CnnEdgeDetection-Keras-Part1.html) / [Jupyter](cnn/CnnEdgeDetection-Keras-Part1.ipynb))
    - Part 2 - Combined horizontal-vertical edge detector using multiple convolutionnal units - coming soon !
- __CNN versus Dense comparison on MNIST__
    - Part 1 - Design and performance comparison ([HTML](cnn/CnnVsDense-Part1.html) / [Jupyter](cnn/CnnVsDense-Part1.ipynb))
    - Part 2 - Visualization with UMAP ([HTML](cnn/CnnVsDense-Part2-Visualization.html) / [Jupyter](cnn/CnnVsDense-Part2-Visualization.ipynb))
    - Part 3 - Resilience to geometric transformations - coming soon!
- **Interpretability**
    - Activation maps on CIFAR-10 ([HTML](cnn/CnnVisualization-1-Activations.html) / [Jupyter](cnn/CnnVisualization-1-Activations.ipynb))
    - Saliency maps on CIFAR-10 ([HTML](cnn/CnnVisualization-2-SaliencyMaps.html) / [Jupyter](cnn/CnnVisualization-2-SaliencyMaps.ipynb))
    - Saliency maps on Imagenet (subset) with ResNet50 ([HTML](cnn/CnnVisualization-3-ResNet.html) / [Jupyter](cnn/CnnVisualization-3-ResNet.ipynb)) (WORK ON GOING)
    - CNN as a graph using NetworkX, extract centrality values ([HTML](cnn/CnnVisualization-5-Graph.html) / [Jupyter](cnn/CnnVisualization-5-Graph.ipynb)) (WORK ON GOING)
- Other CNNs
    - Fashion MNIST CNN with Data Augmentation ([HTML](cnn/CnnMnistFashion-Keras.html) / [Jupyter](cnn/CnnMnistFashion-Keras.ipynb))

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
  
## Data / model sources

### Word embeddings & analysis

- [ConceptNet](http://www.conceptnet.io/)
- [GloVe: Global Vectors for Word Representation - Stanford](https://nlp.stanford.edu/projects/glove/)
- [Opinion Mining, Sentiment Analysis, and Opinion Spam Detection](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)
