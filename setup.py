import sys

try:
    from setuptools import setup, find_packages
    use_setuptools = True
except ImportError:
    from distutils.core import setup
    use_setuptools = False

try:
    with open('README.rst', 'rt') as readme:
        description = '\n' + readme.read()
except IOError:
    # maybe running setup.py from some other dir
    description = ''

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

python_requires = '>=3.6'
install_requires = [
    'xgboost'
]

setup(
    name="xgbtune",
    version='0.1.0',
    url='https://github.com/MainRo/xgbtune.git',
    license='MIT',
    description="Automated XGBoost tunning",
    long_description=description,
    author='Romain Picard',
    author_email='romain.picard@oakbits.com',
    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=pytest_runner,
    tests_require=['pytest>=5.0.1'],
    platforms='any',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3'
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    project_urls={
        'Documentation': 'https://xgbtune.readthedocs.io',
    }
)
