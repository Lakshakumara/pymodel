from setuptools import setup

setup(name='slp',
      packages=['slp'],
      version='0.0.1dev1',
      entry_points={
            'console_scripts': ['slp-cli=slp.cmd:main']
      }
      )