from setuptools import setup, find_packages

setup(
    name="few_shot_neural",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "scikit-learn",
    ],
    description="Few-shot learning metrics for neural data",
    url="https://github.com/KabirDabholkar/few_shot_neural",
    python_requires=">=3.7",
) 