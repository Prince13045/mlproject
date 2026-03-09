from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'


def get_requirement(file_path:str)->List[str]:
    #this fution return the requirement list
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements








setup(
name='mlproject',
version='0.0.1',
author='prince',
author_email='p1304ri@gmail.com',
packages=find_packages(),
install_require=get_requirement('requirement.txt')
)