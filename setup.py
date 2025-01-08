from setuptools import setup, find_packages

setup(
    name='TrueColorHSI',  
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'spectral',
        ' matplotlib',
        ' scipy',
        ' scikit-image',
        ' pysptools',
        'colour-science',
    ],
    author='Fei Zhang',
    author_email='fzhcis@rit.edu',
    description='A package for True Color HSI processing',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fz-rit/TrueColorHSI',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
