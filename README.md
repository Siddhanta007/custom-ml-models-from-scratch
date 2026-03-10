# Custom Machine Learning Models from Scratch

**Algorithms | Optimization Calculus | Object-Oriented Python**

### 🧠 Beyond `import sklearn`
To build robust, production-grade predictive systems, a Data Scientist must understand the mathematical engine driving the models, not just the API. 

This repository contains from-scratch Python/NumPy implementations of core machine learning algorithms. By bypassing high-level libraries, this project demonstrates a deep, applied understanding of loss function optimization, gradient descent calculus, matrix operations, and algorithmic complexity.

### ⚙️ Implemented Architectures
*(Note: These are built using strict Object-Oriented Programming (OOP) principles to mimic standard ML library architectures).*
* **Linear Regression:** (Gradient Descent & Normal Equation implementations)
* **Logistic Regression:** (Sigmoid activation and Cross-Entropy loss optimization)
  
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::Still Under development:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
* **Decision Trees:** (Recursive splitting via Gini Impurity and Information Gain)
* **K-Nearest Neighbors (KNN):** (Custom distance metrics including Euclidean and Manhattan)
* **K-Means Clustering:** (Centroid initialization and iterative reassignment)

### 📐 Mathematical Core
To ensure accurate convergence without relying on black-box libraries, models are optimized using explicit mathematical formulations. 

For instance, the **Linear Regression** implementation minimizes the Mean Squared Error (MSE) cost function:
$$J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2$$

Through vectorized **Batch Gradient Descent**, updating weights iteratively:
$$\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$$

For **Logistic Regression**, the optimization relies on minimizing the Log Loss (Binary Cross-Entropy):
$$J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))]$$

### 🚀 Example Usage (API Design)
The models are designed with a familiar, scikit-learn-style API for seamless training and inference:

```python
import numpy as np
from models.linear_regression import CustomLinearRegression

# 1. Initialize custom model with hyperparameters
model = CustomLinearRegression(learning_rate=0.01, iterations=1000)

# 2. Fit to training data (optimizing via gradient descent)
model.fit(X_train, y_train)

# 3. Generate predictions
predictions = model.predict(X_test)
