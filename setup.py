import os
from distutils.core import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

setup(
    name='tftxtclassify',
    version='0.0.1',
    packages=['tftxtclassify', 'tftxtclassify.classifiers', 'tftxtclassify.search_spaces'],
    # include_package_data=True,
    license='GNU-LGPL-3.0',
    description='A library of tensorflow text classifiers, making it easy to flexibly '
                'reuse classifiers without rewriting a bunch of code.',
    long_description=README,
    url='https://github.com/bnjmacdonald/tftextclassify',
    author='Bobbie Macdonald',
    author_email='bnjmacdonald@gmail.com',
    install_requires = [
        'tensorflow==1.6.0',
        'numpy',
        'scikit-learn',
        'hyperopt',
        'networkx==1.11'
    ],
    zip_safe=False
)
