from setuptools import setup

setup(name='slp',
      packages=['slp'],
      version='1.0.0',
      entry_points={
            'console_scripts': ['sp=slp.cmd:main']
      }
      )