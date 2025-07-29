import numpy as np
import random


"""
this logistic regression model got 99.12% accuracy at a real randomly shuffled data set of breast cancer patients from the sklearn dataset library,
built from the very scratch.
"""



class TrainTheModelError(Exception):
    """Custom exception for accessing model methods before training."""
    def __init__(self, message: str = "You never trained the model!") -> None:
        super().__init__(message)

class UnequalComparison(ValueError):
    def __init__(self, *args):
        super().__init__(*args)




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
        if len(x) != len(y):
            raise UnequalComparison("len(x) is not equal to len(y): len(x) = {len(x)}, len(y) = {len(y)}" )
        self.x,self.std,self.mean = MultipleFeatureLogisticRegression.standardize(np.array(x))
        self.y = np.array(y)
        self.w = np.zeros(self.x.shape[1])
        self.b = 0.0
        self.alpha = alpha
        self.epochs = epochs
        self.__trained = False

    def sigmoid(self, z: float) -> float:
        """Applies the sigmoid activation function."""
        z = np.clip(z, -500, 500)   # this method lets us avoid overflowing cuz z would eventually get veryyyyyy small, less than -500
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def standardize(x: np.ndarray) -> tuple[np.array, float, float]:
        mean = np.mean(x, axis=0)         # mean 
        std = np.std(x, axis=0)           # std 
        std[std == 0] = 1e-8              # avoid division by zero
        x_scaled = (x - mean) / std
        return x_scaled, std, mean

    @staticmethod
    def split(x: np.array, y: list[int], train: int = 0.8, seed: int = 0) -> tuple[np.array, np.array, np.array, np.array]:
        m = len(x) * train
        x,y = MultipleFeatureLogisticRegression.random_shuffle(x,y,seed)
        x_train = x[0:round(m) ]
        x_test  = x[round(m):  ]
        y_train = y[0:round(m)]
        y_test  = y[round(m): ]
        return x_train,y_train, x_test, y_test

    @staticmethod
    def random_shuffle(x,y, seed: int) -> tuple[np.array, np.array]:
        random.seed(seed)
        available = [i for i in range(len(x))]
        random_x = []
        random_y = []
        while available:
            r = random.choice(available)
            i = [j for j in range(len(available)) if available[j] == r][0]
            random_x.append(x[i])
            random_y.append(y[i])
            del available[i]
        return random_x, random_y


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
        x_scaled = (np.array(x) - self.mean) / self.std
        
        z = np.dot(self.w, x_scaled) + self.b
        return int(self.sigmoid(z) >= 0.5)
    


def main() -> None:

    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    
    X = data.data.tolist()  
    y = data.target.tolist()  

    print(f"Number of samples in the dataset: {len(X)}")
    print(f"Number of features in each entry: {len(X[0])}")
    print(f"First sample features: [{X[0][0]}....]")
    print(f"First sample label: {y[0]}") 
    
    x_train,y_train, x_test, y_test = MultipleFeatureLogisticRegression.split(X,y)

    print(f"training samples' length: {len(x_train)}")    
    print(f"first Training samples: [{x_train[0][0]}....]")
    print(f"testing samples' length: {len(x_test)}")
    print(f"first testing sample: [{x_test[0][0]}....]")


    model = MultipleFeatureLogisticRegression(x_train,y_train)
    model.train()

    correct = 0
    for i in range(len(x_test)):
      y_hat = model.predict(x_test[i])
      if y_hat == y_test[i]:
          correct += 1

    accuracy = correct / len(x_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
    
# programmer: Omar Nesr Edin
# GitHub: 7nfya
# instagram: omar.vae
