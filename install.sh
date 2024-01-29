#!/bin/bash

cd esmmodel
chmod 777 ./download.sh
bash ./download.sh

cd ../graph-encoding/multispecies/
java -jar FileUnion.jar ./mu_graph ./graph.emb.npz

cd ../model/
java -jar FileUnion.jar ./uniprot/multispecies ./model.pkl
