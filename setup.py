import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="trainml",
    version="0.0.1",
    description="trainML client SDK and command line utilities",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/trainML/trainml-cli",
    author="trainML",
    author_email="support@trainml.ai",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.7",
        "boto3>=1.16",
        "python-jose[cryptography]>=3.2",
        "requests>=2.25",
    ],
)