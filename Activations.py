import numpy as np

# ---------------------------------------Activation Functions------------------------------------------------
class Tanh:
    """
    Hyperbolic tangent activation function.
    """
    def __call__(self, X):
        """
        Applies the hyperbolic tangent function element-wise.
        """
        self.out = (np.exp(2 * X) - 1) / (np.exp(2 * X) + 1)
        return self.out
    
    def parameters(self):
        """
        Returns an empty list since Tanh has no parameters to maintain consistency with other layers.
        """
        return []

class ReLU:
    """
    Rectified Linear Unit activation function.
    """
    def __call__(self, X):
        """
        Applies the ReLU function element-wise.
        """
        self.out = np.maximum(0, X)
        return self.out
    
    def parameters(self):
        """
        Returns an empty list since ReLU has no parameters to maintain consistency with other layers.
        """
        return []

class LeakyReLU:
    """
    Leaky Rectified Linear Unit activation function.
    """
    def __call__(self, X):
        """
        Applies the Leaky ReLU function element-wise.
        """
        self.out = np.maximum(0.01 * X, X)
        return self.out
    
    def parameters(self):
        """
        Returns an empty list since LeakyReLU has no parameters to maintain consistency with other layers.
        """
        return []
    
class Sigmoid:
    """
    Sigmoid activation function.
    """
    def __call__(self, X):
        """
        Applies the sigmoid function element-wise.
        """
        self.out = 1 / (1 + np.exp(-X))
        return self.out
    
    def parameters(self):
        """
        Returns an empty list since Sigmoid has no parameters to maintain consistency with other layers.
        """
        return []
