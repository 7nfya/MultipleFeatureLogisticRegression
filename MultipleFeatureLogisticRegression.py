import numpy as np

"""
this logistic regression model got 89.46% accuracy at a real data set of breast cancer patients
"""

# to do:
# 1. normalize features


class TrainTheModelError(Exception):
    """Custom exception for accessing model methods before training."""
    def __init__(self, message: str = "You never trained the model!") -> None:
        super().__init__(message)


class MultipleFeatureLogisticRegression:
    """
    A simple implementation of logistic regression with multiple features.
    
    Attributes:
        x (np.ndarray): Feature matrix of shape (m, n)
        y (np.ndarray): Label vector of shape (m,)
        w (np.ndarray): Weight vector of shape (n,)
        b (float): Bias term
        alpha (float): Learning rate
        epochs (int): Number of training iterations
    """
    
    def __init__(
        self,
        x: list[list[float]],
        y: list[float],
        alpha: float = 0.01,
        epochs: int = 1000
    ) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        self.w = np.zeros(self.x.shape[1])
        self.b = 0.0
        self.alpha = alpha
        self.epochs = epochs
        self.__trained = False

    def sigmoid(self, z: float) -> float:
        """Applies the sigmoid activation function."""
        z = np.clip(z, -500, 500)   # this method lets us avoid overflowing cuz z will eventually get veryyyyyy small, less than -500
        return 1 / (1 + np.exp(-z))

    def train(self, ret: bool = False) -> None | tuple[np.ndarray, float]:
        """Trains the model using gradient descent."""
        m = self.x.shape[0]

        for _ in range(self.epochs):
            dw = np.zeros_like(self.w)
            db = 0.0

            for i in range(m):
                xi = self.x[i]
                yi = self.y[i]
                y_pred = self.sigmoid(np.dot(xi, self.w) + self.b)
                error = y_pred - yi

                dw += error * xi
                db += error

            dw /= m
            db /= m

            self.w -= self.alpha * dw
            self.b -= self.alpha * db

        self.__trained = True
        if ret:
            return self.w, self.b

    def predict(self, x: list[float]) -> int:
        """
        Predicts class label (0 or 1) for a given input vector.
        
        Args:
            x (list[float]): Feature list of length equal to training features.
        
        Returns:
            int: 0 or 1 depending on sigmoid threshold.
        """
        if not self.__trained:
            raise TrainTheModelError()

        z = np.dot(self.w, x) + self.b
        return int(self.sigmoid(z) >= 0.5)




def main() -> None:

    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = data.data.tolist()  
    y = data.target.tolist()  

    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(X[0])}")
    print(f"First sample features: {X[0]}")
    print(f"First sample label: {y[0]}") 
    model = MultipleFeatureLogisticRegression(X, y)
    model.train()

    correct = 0
    for i in range(len(X)):
      y_hat = model.predict(X[i])
      if y_hat == y[i]:
          correct += 1

    accuracy = correct / len(X)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
    
# programmer: Omar Nesr Edin
# GitHub: 7nfya
# instagram: omar.vae
