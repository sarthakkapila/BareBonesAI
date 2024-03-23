import numpy as np

# ---------------------------------------EMBEDDING LAYER------------------------------------------------

class Embedding:
    """
    Embedding layer for converting discrete input to dense vectors.
    """
    def __init__(self, in_size, n_emb):
        """
        Initializes the embedding layer with random weights.
        """
        self.C = np.random.randn(in_size, n_emb)
        
    def __call__(self, X):
        """
        Performs the embedding lookup.
        """
        self.emb = self.C[X]
        self.embX = self.emb.view(self.emb.shape[0], -1)
        return self.embX 

# ---------------------------------------COMPOSE MULTIPLE LAYERS------------------------------------------------

class Sequential:
    """
    Sequential model composed of layers executed sequentially.
    """
    def __init__(self, layers):
        """
        Initializes the sequential model with a list of layers.
        """
        self.layers = layers
  
    def __call__(self, X):
        """
        Performs the forward pass through the sequential model.
        """
        for layer in self.layers:
            X = layer(X)
        self.out = X
        return self.out
    
    def parameters(self):
        """
        Returns the learnable parameters of all layers in the sequential model.
        """
        return [p for layer in self.layers for p in layer.parameters()]

# ---------------------------------------RESHAPING LAYER------------------------------------------------

class Flatten:
    """
    Flatten layer to reshape input data.
    """
    def __init__(self, n):
        """
        Initializes the flatten layer with the specified factor.
        
        N - factor to reshape with
        """
        self.N = N
    
    def __call__(self, X):
        """
        Reshapes the input data.
        """
        B, T, C = X.shape
        X = X.view(B, T // self.N, C * self.n)
        if X.shape[1] == 1:
            X = X.squeeze(1)
        self.out = X
        return self.out
  
    def parameters(self):
        
        return []
