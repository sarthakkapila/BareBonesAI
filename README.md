# MiniProp
A mini autograd engine and a neural net library inspired by PyTorch.
Implements backpropagation over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API.  
Potentially useful for educational purposes.

### Installation
To install MiniProp, follow these steps:

1. Clone the MiniProp repository:
   ```bash
   git clone https://github.com/sarthakkapila/MiniProp.git
   ```
2. Navigate to the MiniProp directory:
   ```bash
   cd MiniProp
   ```
3. For a regular installation, run:
    ```python
    python setup.py install
    ```

### Contributing

If you're interested in improving MiniProp or adding new features just create a PR :) .

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/).

```bash
python -m pytest
```

### License

MIT
