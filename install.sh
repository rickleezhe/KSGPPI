#!/bin/bash

cd esmmodel
chmod 777 ./download.sh
bash ./download.sh

cd ..
java -jar FileUnion.jar ./graph-encoding/multispecies/mu_graph ./graph-encoding/multispecies//graph.emb.npz
java -jar FileUnion.jar ./model/multispecies ./model/multispecies/model.pkl
