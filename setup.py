import setuptools

with open('README.md', 'r') as file:
    long_description = file.read()

setuptools.setup(
    name = 'preprocess_dsenthusiast',
    version = '0.0.5',
    author = 'Biju Sasidharan',
    author_email = 'biju.sasidharan@gmail.com',
    description = 'Package to do common text feature engineering and preprocessing',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    packages = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent' 
    ],
    python_requires = '>=3.5'
)