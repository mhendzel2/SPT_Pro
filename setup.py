
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import codecs

# Get the long description from the README file
with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Get package version
with open(os.path.join('spt_analysis', '__init__.py'), 'r') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"').strip("'")
            break

# Main setup configuration
setup(
    name='spt-analysis',
    version=version,
    description='Single-particle tracking analysis package for microscopy data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='SPT Analysis Team',
    author_email='example@example.com',
    url='https://github.com/yourusername/spt-analysis',
    packages=find_packages(include=['spt_analysis', 'spt_analysis.*']),
    include_package_data=True,
    install_requires=[
        'numpy>=1.18.0',
        'scipy>=1.4.0',
        'pandas>=1.0.0',
        'matplotlib>=3.1.0',
        'scikit-image>=0.16.0',
        'scikit-learn>=0.22.0',
        'tifffile>=2020.5.0',
        'h5py>=2.10.0',
        'pyyaml>=5.3.0',
        'networkx>=2.4',
        'seaborn>=0.10.0'
    ],
    extras_require={
        'dev': [
            'pytest>=5.4.0',
            'pytest-cov>=2.8.0',
            'flake8>=3.7.0',
            'black>=19.10b0',
            'sphinx>=3.0.0',
            'sphinx-rtd-theme>=0.5.0'
        ],
        'ui': [
            'pyqt5>=5.15.0',
            'plotly>=4.8.0',
            'ipywidgets>=7.5.0',
            'jupyter>=1.0.0'
        ],
        'deep': [
            'torch>=1.5.0',
            'torchvision>=0.6.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'spt-analysis=spt_analysis.__main__:main',
        ],
        'gui_scripts': [
            'spt-analysis-gui=spt_analysis.__main__:launch_gui',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='microscopy, particle-tracking, single-particle, diffusion, biophysics',
    python_requires='>=3.7',
)
