from setuptools import setup

with open("README.md","r",encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "Backorder",
    version = "0.0.1",
    author = "Jaydeep Dixit",
    description = "Backorder prediction package",
    long_description = long_description,
    url ="https://github.com/Jvdboss7/Backorder-Prediction",
    author_email="jaydeepdixit2@gmail.com",
    packages = ["src"],
    python_requires = ">=3.7",
    install_requires = [
        'dvc',
        'pandas',
        'scikit-learn'
    ]
)