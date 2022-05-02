import setuptools

with open("README.md", "r") as file:
    long_description = file.read()

with open('requirements.txt') as file:
    required = file.read().splitlines()
    
setuptools.setup(
    name="vqaloader",
    version="0.1.1",
    author="Philipp J. Roesch",
    author_email="philipp.roesch@unibw.de",
    description="vqaloader",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phiyodr/vqaloader",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition"
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.csv", "*.json"]},
    install_requires=required
    )
