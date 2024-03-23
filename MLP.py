import numpy as np

class Linear:
    """
    Linear layer with learnable weights and optional bias.
    """
    def __init__(self, fan_in, fan_out, bias=True):
        """
        Initializes the weights and bias of the linear layer.
        """
        self.weights = np.random.randn(fan_in, fan_out)
        self.bias = np.zeros(fan_out) if bias else None
    
    def __call__(self, X):
        """
        Performs the forward pass through the linear layer.
        """
        self.out = np.dot(X, self.weights)
        if self.bias is not None:
            self.out += self.bias
        
        return self.out 

    def parameters(self):
        """
        Returns the learnable parameters of the linear layer.
        """
        return [self.weights] + ([self.bias] if self.bias is not None else [])


