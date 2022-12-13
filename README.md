# ml-math
Mathematical Foundations of Machine Learning!

This repository contains notes and exercises for implementing mathematical foundations of machine learning using [Numpy](https://numpy.org/), [TensorFlow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/), following along with the Udemy course on [Mathematical Foundations of Machine Learning](https://mapbox.udemy.com/course/machine-learning-data-science-foundations-masterclass/) by Dr. Jon Krohn.

## Course Description

Mathematics forms the core of data science and machine learning. Thus, to be the best data scientist you can be, you must have a working understanding of the most relevant math.

Getting started in data science is easy thanks to high-level libraries like Scikit-learn and Keras. But understanding the math behind the algorithms in these libraries opens an infinite number of possibilities up to you. From identifying modeling issues to inventing new and more powerful solutions, understanding the math behind it all can dramatically increase the impact you can make over the course of your career.

Led by deep learning guru Dr. Jon Krohn, this course provides a firm grasp of the mathematics — namely linear algebra and calculus — that underlies machine learning algorithms and data science models.

**Course Sections**

- Linear Algebra Data Structures
- Tensor Operations
- Matrix Properties
- Eigenvectors and Eigenvalues
- Matrix Operations for Machine Learning
- Limits
- Derivatives and Differentiation
- Automatic Differentiation
- Partial-Derivative Calculus
- Integral Calculus

## Jupyter Lab Notebook Setup

All of the Jupyter Lab notebooks in this repository were run using the following steps (assuming docker is already installed):

1. Run minimal Jupyter Lab notebook in docker container:

```bash
docker run -p 8888:8888 -v $(pwd):/home/jovyan/work jupyter/minimal-notebook
```
2. Copy/paste URL token into browser e.g.:
```
http://127.0.0.1:8888/lab?token=b4d93753537af915173d374df05caec6f9cb5808761b3e9b
```
3. Using the Jupyter Lab Terminal session, install the following libraries:
```shell
pip install numpy matplotlib torch tensorflow
```
4. Upload .ipynb file into Jupyter Lab notebook