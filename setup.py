from setuptools import setup, find_packages

setup(
    name='redblue',
    version='0.1',
    packages=find_packages(),
    description='A microservice for positioning, analyzing and visualizing red-blue trains',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        # Classifiers help users find your project
        # Full list: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    # Add more metadata as needed
)
