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
    author="Kabir Dabholkar",
    author_email="kabird@mit.edu",
    description="Few-shot learning metrics for neural data",
    url="https://github.com/KabirDabholkar/few_shot_neural",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 