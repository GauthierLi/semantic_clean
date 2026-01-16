from setuptools import setup, find_packages

setup(
    name="semantic_clean",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'chromadb',
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'scikit-learn',
        'protobuf>=5.0,<7.0'
    ],
    entry_points={
        'console_scripts': [
            'sem_clean=src.main:main',
        ],
    },
)