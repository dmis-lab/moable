import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_hidden_he(layer):
    layer.apply(init_relu)

def init_relu(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight, 2 ** 0.5)

class moable(nn.Module):

    def __init__(self, ecfps):
        super(moable, self).__init__()

        self.ecfps = ecfps
        self.profile_encoder = ProfileEncoder()
        self.drug_encoder = DrugEncoder()

    def forward(self, profile, drug, similarities, cell):

        negatives = torch.multinomial(similarities, 20, False)
        positive_ecfps = self.ecfps(drug)
        negative_ecfps = self.ecfps(negatives)

        profile_embedding = self.profile_encoder(profile,cell)
        positive_drug_embedding = self.drug_encoder(positive_ecfps)
        negative_drug_embedding =  self.drug_encoder(negative_ecfps)

        return profile_embedding, positive_drug_embedding, negative_drug_embedding

class ProfileEncoder(nn.Module):
    
    def __init__(self):
        super(ProfileEncoder, self).__init__()
        self.num_layer = 4
        self.hidden_state = [978, 512, 512, 256, 256]
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.MLP_profile = nn.ModuleList(
            [nn.Linear(self.hidden_state[i], self.hidden_state[i + 1]) for i in range(self.num_layer)])
        init_hidden_he(self.MLP_profile)


    def forward(self, profile, cell):

        for i in range(self.num_layer):

            if i != self.num_layer - 1:
                profile = self.dropout(self.act_func(self.MLP_profile[i](profile)))

            else:
                profile = self.MLP_profile[i](profile)

        return profile

class DrugEncoder(nn.Module):

    def __init__(self):

        super(DrugEncoder, self).__init__()

        self.num_layer = 4
        self.hidden_state = [2048, 2048, 512, 256, 256]
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.MLP_drug = nn.ModuleList(
            [nn.Linear(self.hidden_state[i], self.hidden_state[i + 1]) for i in range(self.num_layer)])
        init_hidden_he(self.MLP_drug)


    def forward(self, drug):

        for i in range(self.num_layer):
            if i != self.num_layer - 1:
                drug = self.dropout(self.act_func(self.MLP_drug[i](drug)))

            else:
                drug = self.MLP_drug[i](drug)
        return drug

