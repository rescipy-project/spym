from setuptools import setup, find_packages

setup(
    name = "rhkpy",
    version = "1.1.0",
    packages = find_packages(),
    # python_requires = '>=3.10',
    install_requires = [
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "xarray>=2023.7.0",
        "holoviews>=1.17.0",
        "hvplot>=0.8.4",
        "bokeh>=3.2.2",
        "panel>=1.2.1"
    ],
    description = "A python package for processing Scanning Tunneling Microscopy (STM) data from RHK, based on the spym project.",
    long_description = open('README.md', 'r').read(),
    long_description_content_type = "text/markdown",
    author = "Peter Nemes-Incze",
    url = "https://github.com/zrbyte/rhkpy",
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
)
