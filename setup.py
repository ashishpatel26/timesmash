from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))

version = {}
with open("timesmash/_version.py") as fp:
    line = fp.read()
    #print(line)
    ls = line.split('.')
    last = str(int(ls[-1].split('"')[0]) + 1) + '"'
    ls[-1] = last
    #print('.'.join(ls))
    ls[-1] = last
    exec(line, version)

with open("timesmash/_version.py", "w") as fp:
	fp.write('.'.join(ls))

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='timesmash',
      version=version['__version__'],
      packages=['timesmash', 'timesmash.bin'],
      keywords='timeseries',      
      install_requires=['pandas', 'numpy', 'scikit-learn'],
      include_package_data=True,
      package_data={
          'bin':
              ['bin/smash',
               'bin/embed',
               'bin/smashmatch',
               'bin/Quantizer',
               'bin/serializer',
               'bin/genESeSS',
               'bin/genESeSS_feature',
               'bin/lsmash',               
               'bin/XgenESeSS'
              ]
      },

      # metadata for PyPI upload
      url='https://github.com/zeroknowledgediscovery/data_smashing_/blob/master/timesmash',

      maintainer_email='virotaru@uchicago.edu',
      maintainer='Victor Rotaru',

      description=('Quantifier of universal similarity amongst arbitrary data'
                   + ' streams without a priori knowledge, features, or'
                   + ' training.'),
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          "Programming Language :: Python :: 3"
      ],
     )


