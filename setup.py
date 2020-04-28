#!/usr/bin/env python3

# 2020 - Centre de Regulacio Genomica (CRG) - All Rights Reserved

from distutils.core import setup

setup(name='anubis',
      description='A package to analyse Tn-seq data',
      author='Samuel Miravet-Verde',
      url='https://github.com/CRG-CNAG/fastqins',
      author_email= 'samuel.miravet@crg.edu',
      version= '0.0.1',
      install_requires= ['re','glob','argparse','subprocess',
                         'numpy','ruffus','Bio','copy','regex',
                         'scipy','pickle','itertools','random',
                         'pandas','ruptures','pylab','seaborn',
                         'matplotlib','operator','joblib',
                         'multiprocessing','collections',
                         'math','sklearn'],
      packages=['anubis'])

# 2020 - Centre de Regulacio Genomica (CRG) - All Rights Reserved

