from setuptools import setup, find_packages

setup(
    name="conv_cp",
    version="1.0",
    packages=find_packages("conv_cp"),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "tensorly",
    ],
)
