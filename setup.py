from setuptools import find_packages, setup

setup(
    name='mytsnelib',
    packages=find_packages(include=['mytsnelib','similarities','utils']),
    version='1.0.0',
    description='A library that implements the T-Sne algorithm',
    author='Victor Gravan Bru',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)