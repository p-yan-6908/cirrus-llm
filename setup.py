from setuptools import setup, find_packages

setup(
    name="cirrus",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
)
