import setuptools

from spym._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spym",
    version=__version__,
    author="Mirco Panighel",
    author_email="mirkopanighel@gmail.com",
    description="A python package for loading and processing Scanning Probe Microscopy (SPM) data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rescipy-project/spym",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "xarray",
        "scipy",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
)
 
