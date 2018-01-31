from setuptools import setup

setup(
    name='cryptoledger',
    packages=['cryptoledger'],
    description='Consolidate transactions history from cryptocurrency exchanges, calculate portfolio and generate a pdf report',
    url='https://github.com/...',
    author='Raphael Ortiz',
    author_email='raphael.ortiz@protonmail.ch',
    version='0.1',
    license='MIT',
    install_requires=['pandas', 'matplotlib', 'seaborn', 'numpy', 'jinja2'],
    include_package_data=True,
        
)
