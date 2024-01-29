# KSGPPI: 
Improving Protein-Protein Interaction Prediction using Protein Language Model and Protein Network Features.

## Pre-requisite:
    - Python3 , Pytorch, Java, Anaconda3
    - networkx
    - esm
    - Linux system 

## Installation:

*Download this repository at  https://github.com/rickleezhe/KSGPPI for academic use. Then, uncompress it and run the following command lines on Linux System.

~~~
  $ cd KSGPPI-main
  $ chmod 777 ./install.sh
  $ ./install.sh
~~~

* If the package cannot work correctly on your computational cluster, you should install the dependencies via running the following commands:

~~~
  $ cd KSGPPI-main
  $ pip install -r requirements.txt
~~~

## Run example
~~~
  $ python predict.py -seq_file1 example/P21346.fa -seq_file2 example/P00955.fa
~~~

## Result

* The prediction result file for the input fasta file (-seq_file1) and fasta file (-seq_file2) can be found in the predict.txt file.
* The prediction result file is one line.  The first protein name, the second protein name, and the predicted probability of their interaction.  For example:
~~~
seq1: P26266, seq2: P00955, Prediction: 0.9984733462333679
~~~

## Tips
This package is only free for academic use.

## References
Jun Hu, Zhe Li, Bing Rao, Maha A. Thafar and Muhammad Arif. Improving Protein-Protein Interaction Prediction using Protein Language Model and Protein Network Features.

## Contact
If you have any questions, comments, or would like to report a bug, please file a Github issue.
