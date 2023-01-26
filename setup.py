import sys
import setuptools

from squigglepy.version import __version__


with open('README.md', 'r') as fh:
    long_description = fh.read()

if sys.version_info < (3, 7):
    raise ValueError('Versions of Python before 3.7 are not supported')

setuptools.setup(
    name='squigglepy',
    version=__version__,
    author='Peter Wildeford',
    author_email='peter@peterhurford.com',
    description=('Squiggle programming language for intuitive probabilistic' +
                 ' estimation features in Python'),
    python_requires='>=3.7',
    install_requires=['msgspec', 'numpy', 'pandas', 'pathos', 'scipy', 'tqdm'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rethinkpriorities/squigglepy',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
 )
