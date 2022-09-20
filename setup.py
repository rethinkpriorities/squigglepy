import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
     name='squigglepy',
     version='0.7',
     author='Peter Hurford',
     author_email='peter@peterhurford.com',
     description=('Squiggle programming language for intuitive probabilistic' +
                  ' estimation features in Python'),
     long_description=long_description,
     long_description_content_type='text/markdown',
     url='https://github.com/peterhurford/squigglepy',
     packages=setuptools.find_packages(),
     classifiers=[
         'Development Status :: 3 - Alpha',
         'Programming Language :: Python :: 3',
         'License :: OSI Approved :: MIT License',
         'Operating System :: OS Independent',
     ],
 )
