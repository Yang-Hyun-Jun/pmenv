from setuptools import find_packages, setup

# Package meta-data
NAME = "pmenv"
DESCRIPTION = "A portfolio rebalancing task environment for reinforcement learning"
URL = "https://github.com/Yang-Hyun-Jun/pmenv.git"
EMAIL = "eppioes@gmail.com"
AUTHOR = "hyunjun"
VERSION = "1.0.1"

# What packages are required for this module to be executed?
REQUIRED = [
    "pandas",
    "numpy",
    "finance-datareader",
    ]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=EMAIL,
    long_description=long_description,
    long_description_content_type = "text/markdown",
    license="MIT",
    packages=find_packages(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=REQUIRED,
    python_requires='>=3.6',
    )

