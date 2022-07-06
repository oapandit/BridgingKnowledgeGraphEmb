# Knowledge graph embeddings to improve Bridging Resolution
This repository contains code for our paper " Integrating knowledge graph embeddings to improve mention representation for bridging anaphora resolution", which appeared in Workshop on Computational Models of Reference, Anaphora and Coreference, COLING 2020.


## Steps to follow
1. The code is based on Python 3.3
2. It uses Support Vector Machines for Ranking implementation given by the authors: https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
3. Download wordnet embeddings from: https://github.com/nlx-group/WordNetEmbeddings
4. Download word2vec, glove embeddings from the authors page.
5. To get embeddings_pp refer: https://zenodo.org/record/1211616#.YsVTv9JByEA
6. All the experiments are present in svm_expts.py, execute that file to get the results.