"""setup.py file for setuptools."""

from setuptools import setup, find_packages

setup(
    name="espora",
    version="0.1.0",
    author="Manas Mahale",
    author_email="manas@pangeabotanica.com",
    description="Espora: Explainable Bioactivity Prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://manasmahale.xyz/posts/espora/",
    packages=find_packages(),
    install_requires=["requests"],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)