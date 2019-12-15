from setuptools import setup, find_packages

setup(name='pyrate',
      version='0.1',
      description='Sports rating system',
      author='John McFarland',
      author_email='mcfarljm@gmail.com',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'sqlalchemy'
      ])
