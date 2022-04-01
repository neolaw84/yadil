from setuptools import setup, find_packages

setup(
    name="yadil",
    version="0.1.0",
    packages=find_packages(include=["yadil", "yadil.*"]),
    install_requires=["bs4", "easydict", "fire"],
    extras_require={
        # 'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    },
)
