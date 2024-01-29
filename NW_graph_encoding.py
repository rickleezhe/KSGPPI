import sys
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-species", "--species", default='multispecies')
parser.add_argument("-seq_file1", "--seq_file1")
parser.add_argument("-seq_file2", "--seq_file2")
parser.add_argument("-dv", "--device", default='cuda')
args = parser.parse_args()



with open('./graph-encoding/'+str(args.species)+'/'+str(args.species)+'.protein.fa.tsv','r') as ff:
    fasta_lines = ff.readlines()
fasta = {}
for j in fasta_lines:
    fasta[j.strip().split('\t')[0]] = j.strip().split('\t')[1]
# print(fasta)
print(len(fasta))

def loadFasta(fasta):
    with open(fasta, 'r') as f:
        lines = f.readlines()
    ans = {}
    name = ''
    seq_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if 0 < len(name):
                ans[name] = "".join(seq_list)
            name = line[1:]
            seq_list = []
        else:
            seq_list.append(line)
    if 0 < seq_list.__len__():
        ans[name] = "".join(seq_list)
    return ans

def NWalign(protein_A, protein_B):
    cmd = r"java -jar ./graph-encoding/NWAlign.jar "+protein_A+" "+protein_B+" "+str(3)
    result = os.popen(cmd, "r")
    result = result.readlines()[5].strip()
    seqIdentity = np.array(result.split("=")[1][:5], dtype=float)
    return seqIdentity

# if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         seq_file1 = sys.argv[1]
#         seq_file2 = sys.argv[2]
#     pro1 = loadFasta(seq_file1)
#     pro2 = loadFasta(seq_file2)
#
#     name1, seq1 = next(iter(pro1.items()))
#     name2, seq2 = next(iter(pro2.items()))
#     # for row in pro1:
#     #     name1 = row
#     #     seq1 = str(pro1[row])
#     #
#     # for row2 in pro2:
#     #     name2 = row2
#     #     seq2 = str(pro2[row2])
#     pro1_string_score ={}
#     for one in fasta:
#         score1 = NWalign(seq1, fasta[one])
#         print(name1,one,score1)
#         pro1_string_score[one] = score1
#         if score1 == 1.000:
#             break
#
#     max_letter1 = max(pro1_string_score, key=pro1_string_score.get)
#     print(f"The letter with the maximum value is: {max_letter1}")
#
#     pro2_string_score ={}
#     for one in fasta:
#         score2 = NWalign(seq2, fasta[one])
#         print(name2,one,score2)
#         pro2_string_score[one] = score2
#         if score2 == 1.000:
#             break
#
#     max_letter2 = max(pro2_string_score, key=pro2_string_score.get)
#     print(f"The letter with the maximum value is: {max_letter2}")
#
#
#
#     graph_emb = np.load('./graph.emb.npz')
#     print(graph_emb[max_letter1])
#     print(graph_emb[max_letter1].shape)
#
#     print(graph_emb[max_letter2])
#     print(graph_emb[max_letter2].shape)1.shape)
#     print(EKS_coding2.shape)
#     print(EKS_coding1)
#     print(EKS_coding2)
