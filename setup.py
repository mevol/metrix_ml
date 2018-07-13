from setuptools import setup

setup(
    name = "metrix_ml",
    version = "0.0.1",
    author = "Melanie Vollmar",
    author_email = "melanie.vollmar@diamond.ac.uk",
    description = "A package that classifies maps",
    license = "BSD",
    keywords = "awesome python package",
    packages=[
      'metrix_ml', 
      'tests'
    ],
    scripts=[
    ],
    install_requires=[
      'pytest',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
)