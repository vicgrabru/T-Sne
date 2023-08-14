from setuptools import find_packages, setup

setup(
    name='mytsnelib',
    packages=find_packages(),
    version='0.1.0',
    description='A library that implements the T-Sne algorithm',
    author='Victor Gravan Bru',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)