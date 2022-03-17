from setuptools import setup
from setuptools import find_packages

setup(name='M-GCN',
      description='Molecular Subtyping of Cancer based on Robust Graph Neural Network and Multi-omics Data Integration',
      download_url='https://github.com/yinchaoyi/M-GCN',
      install_requires=['numpy',
                        'torch',
                        'pandas',
                        'scikit-learn',
                        'pyHSICLasso',
                        'scipy',
                        'argparse',
                        'random',
                        'time',
                        'numpy',
                        'math'
                        ],
      package_data={'M-GCN': ['README.md']},
      packages=find_packages())
