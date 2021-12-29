# Introduction to Ray Core Design Patterns and APIs.

<img src="images/ray-logo.png" height="50%" width="50%">


This is an introduction to basic Ray patterns for distributing computing. In this tutorial, we will cover at least three basic Ray patterns and its respective Ray Core APIs. 

 * Remote Stateless Ray Tasks
 * Remote Stateful Ray Actors
 * Remote ObectRefs as Futures

By no means all the Ray patterns are covered here. We recommend that you follow the references for [advanced patterns and antipatterns](https://docs.ray.io/en/latest/ray-design-patterns/index.html) if you want to use Ray to write your own ML-based libraries or want to take existing Python single-process or single-node multi-core applications and covert them into distributed multi-core, multi-node processes on a Ray cluster.

Knowing these Ray patterns and anti-patterns will guide you in writing effective and robust distributed applications using the Ray framework and its recommended usage of Ray APIs.

### Instructions to get started

We assume that you have a `conda` installed.

 1. `source env.sh ray-core 3.8.10` 
 3. `git clone git@github.com:dmatrix/ray-core-tutorial.git`
 4. `cd` to `<cloned_dir>`
 5. `jupyter lab`
 
 Enjoy Ray!
 
 Jules

