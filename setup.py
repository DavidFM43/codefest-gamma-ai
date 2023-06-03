from setuptools import find_packages, setup
setup(
    name='gamma',
    packages=find_packages(include=['gamma']),
    version='0.1.0',
    description='Library for the Codefest Ad Astra 2023.', 
    author='Gamma AI',
    license='MIT',
    install_requires=["flair", "newspaper3k"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)
