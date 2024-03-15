import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MiniTorch",
    version="0.1.0",
    author="Sarthak Kapila",
    author_email="sarthakkapila1@gmail.com",
    description="Minimalist deep-learning framework with a similar API to PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karpathy/MiniTorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)