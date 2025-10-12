from setuptools import setup, find_packages

setup(
    name="fraudguard-shared",
    version="1.0.0",
    description="Shared data models and utilities for FraudGuard 360",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
    ],
    python_requires=">=3.9",
)