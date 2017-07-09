from setuptools import setup, find_packages


setup(
    name='amii-tf-nn',
    version='0.0.1',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'future == 0.15.2',
        'setuptools >= 20.2.2',
        'pyyaml == 3.12',
        # tensorflow or tensorflow-gpu v1.2
    ],
    tests_require=['pytest', 'pytest-cov'],
    setup_requires=['pytest-runner'],
    classifiers=[
        'License :: OSI Approved :: MIT License'
    ],
)
