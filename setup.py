from setuptools import find_packages, setup

setup(
    name="cmp4501",
    version="0.1",
    packages=find_packages(),
    install_requires=["pytest", "pytest-cov"],
)
