# -*- coding: UTF-8 -*-
import esm
import torch.nn as nn
import torch
import numpy as np
import sys
import math
from torch.autograd import Variable
from itertools import product

class ESMFineTune(nn.Module):
    def __init__(self, esm2_layer=0):
        super(ESMFineTune, self).__init__()
        self.esm2_layer = esm2_layer
#        self.Esm2, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        local_path = './esmmodel/esm2_t33_650M_UR50D.pt'  
        self.Esm2, self.alphabet = esm.pretrained.load_model_and_alphabet_local(local_path)
        self.tokenizer1 = self.alphabet.get_batch_converter()
        self.esm2_freeze()


    def esm2_freeze(self):
        esm2_all_layer_dict = {}
        esm2_num = 0
        esm2_unfreeze_layer_dict = {}

        for name, param in self.Esm2.named_parameters():
            param.requires_grad = False
        for name, param in self.Esm2.named_parameters():
            if "layers" in name:
                key = name[0: name.find('.', 7)]
                if key in esm2_all_layer_dict.keys():
                    if key in name:
                        esm2_all_layer_dict[key].append(name)
                else:
                    esm2_all_layer_dict[key] = []
                    if key in name:
                        esm2_all_layer_dict[key].append(name)
        for i, name in enumerate(esm2_all_layer_dict):
            if len(esm2_all_layer_dict) - self.esm2_layer < 0:
                exit("层数溢出")
            if esm2_num >= len(esm2_all_layer_dict) - self.esm2_layer:
                esm2_unfreeze_layer_dict[name] = esm2_all_layer_dict[name]
            esm2_num += 1

        not_freeze_param_nums_esm2 = 0
        for name, param in self.Esm2.named_parameters():
            if name[0: name.find('.', 7)] in esm2_unfreeze_layer_dict.keys():
                param.requires_grad = True
                not_freeze_param_nums_esm2 += param.numel()

    def forward(self, x):
        _, _, tokens = self.tokenizer1(x)
        with torch.no_grad():
            representations = self.Esm2(tokens, repr_layers=[33], return_contacts=True)["representations"][33]
        mean_representations = representations[:, 1: len(x[0][1]) + 1].squeeze().numpy()
        return mean_representations



class CKSAAP(nn.Module):
    def __init__(self, is_use_position=False, position_d_model=None):
        super(CKSAAP, self).__init__()

        AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        DP = list(product(AA, AA))
        # print(DP) #[('A', 'A'), ('A', 'C'), ..., ('Y', 'W'), ('Y', 'Y')]
        # print(len(DP)) #400
        self.DP_list = []
        for i in DP:
            self.DP_list.append(str(i[0]) + str(i[1]))
        # print(self.DP_list)  #['AA', 'AC', ..., 'YW', 'YY']

        self.position_func = None
        self.position_d_model = position_d_model
        if is_use_position:
            if None is position_d_model:
                self.position_d_model = 16
            self.position_func = PositionalEncoding(d_model=self.position_d_model)

    def returnCKSAAPcode(self, query_seq, k=3):
        code_final = []
        for turns in range(k + 1):
            DP_dic = {}
            code = []
            code_order = []
            for i in self.DP_list:
                DP_dic[i] = 0
            for i in range(len(query_seq) - turns - 1):
                tmp_dp_1 = query_seq[i]
                tmp_dp_2 = query_seq[i + turns + 1]
                tmp_dp = tmp_dp_1 + tmp_dp_2
                if tmp_dp in DP_dic.keys():
                    DP_dic[tmp_dp] += 1
                else:
                    DP_dic[tmp_dp] = 1
            for i, j in DP_dic.items():
                code.append(j / (len(query_seq) - turns - 1))
            for i in self.DP_list:
                code_order.append(code[self.DP_list.index(i)])
            code_final += code

        code_final = torch.FloatTensor(code_final)
        code_final = code_final.view(k+1, 20, 20)
        return code_final

    def return_CKSAAP_Emb_code(self, query_seq, emb, k=3, is_shape_for_3d=False):
        """
        :param is_shape_for_3d:
        :param query_seq: L
        :param emb: [L, D] tensor
        :param k:
        :return:
        """
        code_final = []
        for turns in range(k + 1):
            DP_dic = {}
            code = []
            code_order = []
            for i in self.DP_list: ##['AA', 'AC', ..., 'YW', 'YY']
                DP_dic[i] = torch.zeros(emb.size(-1))
            # print(DP_dic) #{'AA': tensor([0., 0., 0.,  ..., 0., 0., 0.]),..., 'YY': tensor([0., 0., 0.,  ..., 0., 0., 0.])}  #1280
            for i in range(len(query_seq) - turns - 1):
                tmp_dp_1 = query_seq[i]
                tmp_dp_2 = query_seq[i + turns + 1]
                tmp_emb_1 = emb[i]
                tmp_emb_2 = emb[i + turns + 1]
                tmp_emb = 0.5 * (tmp_emb_1 + tmp_emb_2)

                tmp_dp = tmp_dp_1 + tmp_dp_2
                if tmp_dp in DP_dic.keys():
                    DP_dic[tmp_dp] += tmp_emb
                else:
                    DP_dic[tmp_dp] = tmp_emb
            for i, j in DP_dic.items():
                code.append(j / (len(query_seq) - turns - 1))
            for i in self.DP_list:
                code_order.append(code[self.DP_list.index(i)])
            code_final += code

        code_final = torch.stack(code_final)
        code_final = code_final.view(k+1, 20, 20, -1)

        if is_shape_for_3d:
            k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = code_final.size()
            code_final = code_final.permute(0, 3, 1, 2).contiguous().\
                view(k_plus_one * position_posi_emb_size, aa_num_1, aa_num_2)
        return code_final

    def return_CKSAAP_position_code(self, query_seq, k=3):
        """
        :param query_seq: L
        :param embs: [L, D] tensor
        :param k:
        :return: [(k+1)*position_posi_emb_size, 20, 20]
        """
        posi_emb = self.position_func(
            torch.zeros(1, len(query_seq), self.position_d_model)
        ).squeeze(0)

        # [(k+1), 20, 20, position_posi_emb_size]
        emb = self.return_CKSAAP_Emb_code(query_seq, posi_emb, k)

        # [(k+1), 20, 20, position_posi_emb_size] --> [(k+1)*position_posi_emb_size, 20, 20]
        k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = emb.size()
        emb = emb.permute(0, 3, 1, 2).contiguous().view(k_plus_one*position_posi_emb_size, aa_num_1, aa_num_2)
        return emb

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

# if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         seq_file1 = sys.argv[1]
#         seq_file2 = sys.argv[2]
#     pro1 = loadFasta(seq_file1)
#     pro2 = loadFasta(seq_file2)
#     for row in pro1:
#         name1 = row
#         seq1 = str(pro1[row])
#
#     for row2 in pro2:
#         name2 = row2
#         seq2 = str(pro2[row2])
#     esm_ = ESMFineTune(esm2_layer=0)  # .to(device)
#
#     esm_out1 = esm_([(name1,seq1)])
#     esm_out2 = esm_([(name2,seq2)])
#
#
#     CKSAAP_ = CKSAAP()
#     EKS_coding1 = CKSAAP_.return_CKSAAP_Emb_code(seq1, torch.tensor(esm_out1), is_shape_for_3d=True)
#     EKS_coding2 = CKSAAP_.return_CKSAAP_Emb_code(seq2, torch.tensor(esm_out2), is_shape_for_3d=True)
#
# #    np.save('Data/Yeast/PIPR-cut/Ks-coding/'+name+'.npy', textembed_1)
#     print(EKS_coding1.shape)
#     print(EKS_coding2.shape)
#     print(EKS_coding1)
#     print(EKS_coding2)





