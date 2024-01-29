from ESM2_CKSAAPEncoding import *
from NW_graph_encoding import *
import torch
from TAGlayer import *
import argparse

def predict_single_data(model, G1, G2, dmap1, dmap2):
    model.eval()
    with torch.no_grad():
        output = model(G1, G2, dmap1, dmap2)
    return output.item()  # Assuming the output is a single value

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-species", "--species", default='multispecies')
    parser.add_argument("-seq_file1", "--seq_file1")
    parser.add_argument("-seq_file2", "--seq_file2")
    parser.add_argument("-dv", "--device", default='cuda')
    args = parser.parse_args()

    # if len(sys.argv) > 1:
    #     seq_file1 = sys.argv[1]
    #     seq_file2 = sys.argv[2]
    device = args.device
    pro1 = loadFasta(args.seq_file1)
    pro2 = loadFasta(args.seq_file2)
    name1, seq1 = next(iter(pro1.items()))
    name2, seq2 = next(iter(pro2.items()))

    esm_ = ESMFineTune(esm2_layer=0)  # .to(device)
    esm_out1 = esm_([(name1,seq1)])
    esm_out2 = esm_([(name2,seq2)])

    CKSAAP_ = CKSAAP()
    EKS_coding1 = CKSAAP_.return_CKSAAP_Emb_code(seq1, torch.tensor(esm_out1), is_shape_for_3d=True)
    EKS_coding2 = CKSAAP_.return_CKSAAP_Emb_code(seq2, torch.tensor(esm_out2), is_shape_for_3d=True)

    # print(EKS_coding1.shape)
    # print(EKS_coding2.shape)
    # print(EKS_coding1)
    # print(EKS_coding2)

#    print(type(EKS_coding2))

    max_letter1 = '362663.ECP_0619'
    max_letter2 = '362663.ECP_1666'

#    pro1_string_score ={}
#    for one in fasta:
#        score1 = NWalign(seq1, fasta[one])
#        # print(name1,one,score1)
#        pro1_string_score[one] = score1
#        if score1 == 1.000:
#            break
#    pro2_string_score ={}
#    for one in fasta:
#        score2 = NWalign(seq2, fasta[one])
#        # print(name2,one,score2)
#        pro2_string_score[one] = score2
#        if score2 == 1.000:
#            break
#
#    max_letter1 = max(pro1_string_score, key=pro1_string_score.get)
#    max_letter2 = max(pro2_string_score, key=pro2_string_score.get)

#    graph_emb = np.load('./graph-encoding/graph.emb.npz')
    graph_emb = np.load('./graph-encoding/'+str(args.species)+'/graph.emb.npz')


    dmap1=EKS_coding1.unsqueeze(0).to(device)
    dmap2=EKS_coding2.unsqueeze(0).to(device)
    G1_ = torch.tensor(graph_emb[max_letter1]).float()
    G2_ = torch.tensor(graph_emb[max_letter2]).float()
    G1 = G1_.unsqueeze(0).to(device)
    G2 = G2_.unsqueeze(0).to(device)


    model = KSGPPI().to(device)
    model.load_state_dict(torch.load('model/'+str(args.species)+'/model.pkl'))

    prediction = predict_single_data(model, G1, G2, dmap1, dmap2)
    print(f'seq1: {name1},seq2: {name2}, Prediction: {prediction}')

    with open('predict.txt', 'a') as file:
        file.write(f'seq1: {name1}, seq2: {name2}, Prediction: {prediction}\n')