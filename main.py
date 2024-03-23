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

# ---------------------------------------NORMALIZATION------------------------------------------------

class BatchNorm1d:
    """
    BatchNorm1d normalization layer.

        eps (float): Small value added to the denominator for numerical stability.
        momentum (float): Momentum factor for updating running mean and variance.
        alpha & beta: hyperparameters

        Xhat = (X - mean(X))/sqrt(var + eps)
        Y = gammaXhat + beta
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
  
    def __call__(self, X):

        if self.training:
        if X.ndim == 2:
            dim = 0
        elif X.ndim == 3:
            dim = (0,1)
        Xmean = X.mean(dim, keepdim=True) # batch mean
        Xvar = X.var(dim, keepdim=True) # batch variance
        else:
        Xmean = self.running_mean
        Xvar = self.running_var
        Xhat = (X - Xmean) / torch.sqrt(Xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * Xhat + self.beta

        if self.training:
        with torch.no_grad():
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
  
    def parameters(self):
        return [self.gamma, self.beta]