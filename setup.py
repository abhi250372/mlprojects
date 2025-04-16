from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path:str) -> List[str]:
    """
    This function returns a list of requirements from the given file path.
    """
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
name="Prophet ML Time Series Forecasting",
version="0.1.0",
author="Abhishek Pawar",
author_email="abhirvp1234@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)