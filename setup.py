from setuptools import setup, find_packages

setup(
    name="ceres_infer",
    version="1.0",
    author="Boyang Zhao",
    description='CERES inference',
    long_description=open('README.md').read(),
    package_dir={"": "src"},
    packages=find_packages("ceres_infer"),
    include_package_data=True,
    zip_safe=False,
    install_requires=open('requirements.txt').read().strip().split('\n')
)
