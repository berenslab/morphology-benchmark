#!/bin/bash

#export PYTHONPATH=$PYTHONPATH:~/.ipython/profile_default/startup/

# Neuron 
#printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.cell import Neuron; Neuron().populate(\\"ds_id=1 and reduction_id=0\\",reserve_jobs=True)";\n%.0s' {1..10}| parallel -j 10 --no-notice

# morphometry
#printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.morphometry import Morphometry; Morphometry().populate(\\"ds_id<6\\",reserve_jobs=True)";\n%.0s' {1..10}| parallel -j 10 --no-notice

# morphometric Distributions
#printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.morphometry import MorphometricDistribution; MorphometricDistribution().populate(\\"ds_id<7\\",reserve_jobs=True)";\n%.0s' {1..10}| parallel -j 10 --no-notice

# Persistence 
#printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.persistence import Persistence; Persistence().populate(\\"ds_id<6\\",reserve_jobs=True)";\n%.0s' {1..10}| parallel -j 10 --no-notice

# density map
#printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.density import SampledDensityMap; SampledDensityMap().populate(\\"ds_id<6\\",reserve_jobs=True)";\n%.0s' {1..10}| parallel -j 10 --no-notice

# Feature Maps
#printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.features import FeatureMap; FeatureMap().populate(\\"ds_id <6 and  reduction_id=0\\",reserve_jobs=True)";\n%.0s' {1..10}| parallel -j 10 --no-notice


# imbalanced Pairwise Classification 
printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.classification import PairwiseClassificationImbalanced; PairwiseClassificationImbalanced().populate(\\" (classifier_id=1 or classifier_id>4) and reduction_id=0 and (ds_id !=2 and ds_id <6) and (statistic_id > 110)\\",reserve_jobs=True)"; \n%.0s' {1..10}| parallel -j 10 --no-notice


# imbalanced Multiclass Classification
#printf 'python3 -c "import datajoint as dj; dj.config.load(\\"./utils/dj_mysql_conf.json\\"); c = dj.conn(); from schemata.classification import MultiClassClassificationImbalanced; MultiClassClassificationImbalanced().populate(\\"classifier_id>4 and (ds_id != 2 and ds_id < 6) and reduction_id=0\\",reserve_jobs=True)"; \n%.0s' {1..10}| parallel -j 10 --no-notice
