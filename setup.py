from setuptools import setup
import os
import subprocess

path_requirements = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
print('Installing requirements...')
subprocess.run(f'python -m pip install -r {path_requirements}', shell=True)

setup(
    name='pointnet',
    version='1.0',
    packages=['pointnet']
)
