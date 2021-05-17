from rdkit.Chem import AllChem, Lipinski, Descriptors, Crippen
from rdkit import Chem, DataStructs
import torch
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gseapy as gp
import model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.DrugEncoder()
model.load_state_dict(torch.load('model/moable.pth'))
model.to(device)
model.eval()

def smiles2fp(smilesstr):
    mol = Chem.MolFromSmiles(smilesstr)
    fp_obj = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048,
                                                   useChirality=True)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_obj, arr)

    return arr

def drug_embeddings(drug_dict):

    result_dict = dict()

    for key in drug_dict:
        smiles = drug_dict[key]
        ecfp = torch.from_numpy(smiles2fp(smiles)).to(device)
        ecfp = ecfp.reshape(1,-1)
        embedding = model(ecfp.float()).cpu().detach().numpy().flatten()
        magnitude = np.linalg.norm(embedding)
        embedding = embedding / magnitude
        result_dict[key] = embedding

    return result_dict

with open('data/input/example_input_smiles.pkl', 'rb') as f: 
    drug_dict = pickle.load(f)

CP_embedding_dict = drug_embeddings(drug_dict)

GP_sig_df = pd.read_pickle('data/GP_sig_df.pkl')
GP_embedding_dict = pd.read_pickle('data/GP_embedding_dict.pkl')

similarities = cosine_similarity(list(CP_embedding_dict.values()),list(GP_embedding_dict.values()))

i = 0

for drug in CP_embedding_dict:
    connectivity_score_df = pd.DataFrame({'sig_id':list(GP_embedding_dict.keys()),'score':similarities[i]})
    connectivity_score_df = connectivity_score_df.merge(GP_sig_df[['sig_id','cmap_name']])
    connectivity_score_df = connectivity_score_df.groupby(['cmap_name']).max().sort_values(by = ['score'], ascending = False).reset_index()

    pathway_res = gp.prerank(rnk=connectivity_score_df[['cmap_name','score']], 
        gene_sets='KEGG_2019_Human',
            processes=100,
            permutation_num=1000, 
            no_plot = True,
        format='png', seed=10)

    pathway_res.res2d.to_csv('data/output/{}.csv'.format(drug))
    i+=1

