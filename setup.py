from setuptools import setup, find_packages

setup(
    name="nectar",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "matplotlib",
        "tqdm",
        "scikit-learn",
        "scipy",
        "seaborn",
        "dill",
        "openpyxl"
        ],
)
