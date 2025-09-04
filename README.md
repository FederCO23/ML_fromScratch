# ML_fromScratch <img src="./images/MLrobot.PNG" align="right" width="250" />
A collection of foundational Machine Learning algorithms built from scratch to demystify the core concepts behind them.
<br><br>
<br><br>
<br><br>
## Context

This series of notebooks and libraries aims to build foundational Machine Learning models from scratch using **NumPy** and other basic libraries, without relying on high-level ML frameworks. The goal is to strengthen understanding by implementing each model step by step.

The project was originally structured into four parts, guided by my mentor **Artem Yankov** (@Google) and inspired by the **Machine Learning Specializations** by **Andrew Ng**. I’ve since continued expanding the repository by adding more models, such as **Naive Bayes**, to enrich the collection.

Notably, in **Part 3 – Neural Networks**, we include detailed mathematical derivations of forward and backward propagation, providing a deeper understanding of how these models learn.


## Summary

**Core Parts**
- [CP_part1](./notebooks/ML_firstSteps_P1.ipynb): **Linear Regression** (Simple LR, Multiple LR and Polynomial LR)
- [CP_part2](./notebooks/ML_firstSteps_P2.ipynb): **Logistic Regression**
- [CP_part3](./notebooks/ML_firstSteps_P3.ipynb): **Neural Networks** (Regression & Classification)
- [CP_part4](./notebooks/ML_firstSteps_P4.ipynb): **Neural Networks applied to MNIST dataset** (3 layers NN using a custom-built library)
- [CP_part5](./notebooks/ML_fromScratch_P5.ipynb): **Naive Bayes**
- [CP_part6](./notebooks/ML_fromScratch_P6.ipynb): **KNN & LSH**

**Word Embeddings**
- [WE_part1](./notebooks/ML_fromScratch_WE_P1.ipynb): custom **Word2Vec** (SGNS) implementation

**Intuitions**
- [I_part1](./notebooks/ML_intuitions_P1.ipynb):  **PCA** insights, visuals and geometric intuition.


## Highlights

### A. The relevance of Scaling

Beyond implementing algorithms from scratch, we uncovered key insights during the process. One is the crucial role of feature scaling in optimization and convergence. The other highlights how neural networks evolve during training, shedding some light on parameter updates and learning dynamics.


#### Linear Regression:
<img src="./images/Normalization.png" width="850" />

Feature scaling significantly accelerates convergence. The cost function `J(w,b)` reaches a lower value more efficiently, even when starting from zero-initialized weights.


#### Logistic Regression

<img src="./images/LogR-NO-Scaling.png" width="400" /> <img src="./images/LogR-WITH-Scaling.png" width="408" />

Observe the difference of using Logistic Regression without(left) and with(right) scaling. Scaling ensures better convergence and prevents numerical instability.



### B. Tracking Neural Network Weight changes during Gradient Descent


<img src="./images/ParamsChange.png" width="500" />

This visualization demonstrates how NN parameters (weights) evolve over sucessive gradient descent steps.

After the storm comes the calm...🌤️