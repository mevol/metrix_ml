from setuptools import setup

setup(
    name = "metrix_ml",
    version = "0.0.1",
    author = "Melanie Vollmar",
    author_email = "melanie.vollmar@diamond.ac.uk",
    description = "A package which uses crystallographic statistics and metrics \n"
    							"to make predictions about the likelihood of solving a structure",
    license = "BSD",
    keywords = "awesome python package",
    packages=[
      'metrix_ml'
    ],
    scripts=[
      'bin/decisiontree_gridsearch',
      'bin/decisiontree_randomsearch',
      'bin/randomforest_gridsearch',
      'bin/decisiontree_bag_gridsearch',
      'bin/randomforest_randomsearch'
    ],
    install_requires=[
      'matplotlib',
      'pandas',
      'pytest',
      'sklearn'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
)
