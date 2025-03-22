from setuptools import setup, find_packages

setup(
    name='kitml',
    version='0.1',
    packages=find_packages(),
     install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'matplotlib>=3.1.0',
        'scikit-learn>=0.22',
        'tqdm',
        'ABC'
    ],
)
