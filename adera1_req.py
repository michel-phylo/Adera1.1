import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
dependencies = [

  'keybert>=0.5.0',
  'metapub>=0.5.5',
  'fstrings>= 0.1.0',
  'textract>=1.6.5',
  'tika>=1.24',
   'tika-app>= 1.5.0',
   'pandas>= 1.4.1', 
   'tensorflow>=1.7',
   'tensorflow-hub>= 0.12.0',
    'seaborn>=0.11.2',
    'chainer>=7.8.1',
    'matplotlib>=3.5.1',
    'numpy>= 1.20.2',
    'keras>=2.8.0'
]

# here, if a dependency is not met, a DistributionNotFound or VersionConflict
# exception is thrown. 
pkg_resources.require(dependencies)
print("Macrious")
