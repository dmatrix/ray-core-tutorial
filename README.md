# Introduction to Ray Ecosystem for Distributed Applications 

<img src="images/PyCon2022_Logo.png" height="50%" width="80%">

Welcome to the tutorial at PyCon US 2022 in Salt Lake City

<img src="images/ray-logo.png" height="50%" width="50%">


This is a gentle introduction to basic Ray programming patterns and APIs for distributing computing. In this tutorial, we will cover at least three basic Ray patterns and its respective Ray Core APIs. 

 * Remote Stateless Ray Tasks
 * Remote Stateful Ray Actors
 * Remote ObectRefs as Futures
 
Additionally, a brief introductino to two of the Ray native libraries:
 * Introduction to Ray Tune and Ray Serve

By no means all the Ray patterns and APIs are covered here. We recommend that you follow the references for [advanced patterns and antipatterns](https://docs.ray.io/en/latest/ray-design-patterns/index.html) if you want to use Ray to write your own ML-based libraries or want to take existing Python single-process or single-node multi-core applications and covert them into distributed multi-core, multi-node processes on a Ray cluster.

Knowing these Ray patterns and anti-patterns will guide you in writing effective and robust distributed applications using the Ray framework and its recommended usage of Ray APIs.

Additoinaly, we'll briefly examine how to use Tune APIs to train and tune your model, followed by an introduction
to Ray Serve for deploying and serving models.

### Prerequisite knowledge ###

Some prior experience with Python and Jupyter notebooks will be helpful, but we'll explain most details as we go if you haven't used notebooks before. Knowledge of basic machine learning concepts, including hyperparameters, model serving, and principles of distributed computing is helpful, 
but not required.

All exercises can be done on your laptop, preferably running a Linux or macOS, using all its cores. Because you won’t have access to Ray clusters, we have to run Ray locally and parallelize all your tasks on all your cores.

Python 3.7+ is required on your laptop, and some minimal installation of quick python packages using conda and pip.

### Instructions to get started

We assume that you have a `conda` installed.

 1. `conda create -n ray-core-tutorial python=3.8`
 2. `conda activate ray-core-tutorial`
 3. `git clone git@github.com:dmatrix/ray-core-tutorial.git`
 4. `cd` to <cloned_dir>
 5. `python3 -m pip install -r requirements.txt`
 6. `python3 -m ipykernel install`
 7. `jupyter lab`
 
 If you are using **Apple M1 laptop** follow the following instructions:
 
 1. `conda create -n ray-core-tutorial-testing python=3.8`
 2. `conda activate ray-core-tutorial-testing`
 3. `conda install grpcio`
 4. `python3 -m pip install -r requirements.txt`
 5. `python3 -m ipykernel install`
 6. `conda install jupyterlab`
 7. `jupyter lab`
 
Let's have fun with Ray @ PyCon US 2022!
 
Jules
