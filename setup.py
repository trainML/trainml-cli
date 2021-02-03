import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()


setup(
    name="trainml",
    version="0.1.0",
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
        "aiohttp",
        "boto3",
        "python-jose[cryptography]",
        "requests",
    ],
)