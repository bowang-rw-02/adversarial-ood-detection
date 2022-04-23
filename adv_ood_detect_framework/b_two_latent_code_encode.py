import torch
import torch.nn as nn
import numpy as np
import pickle
from keras_preprocessing import sequence
import os
import json


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, weights_matrix, n_layers):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.embedding.weight = nn.Parameter(torch.tensor(weights_matrix, dtype=torch.float32))

        self.LSTM = nn.LSTM(input_size= emb_dim,
                            hidden_size= hid_dim,
                            num_layers= n_layers,
                            bidirectional= True,
                            batch_first= False)
        self.fc_h = nn.Linear(hid_dim*2, hid_dim)
        self.fc_c = nn.Linear(hid_dim*2, hid_dim)

    def forward(self, src):
        # src = [src sent len, batch size]
        embedded = self.embedding(src)
        # embedded = [src sent len, batch size, emb dim]
        outputs, (hidden, cell) = self.LSTM(embedded)
        # outputs = [src len, batch size, hid dim * 2directions]
        # hidden = [n layers=1 * n dire = 2, batch size, hid dim]

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc_h(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        hidden = hidden.unsqueeze(0)
        # hidden = [batch size, hid dim]
        cell = torch.tanh(self.fc_c(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1)))
        cell = cell.unsqueeze(0)
        return hidden, cell

def encode_data_to_vector():
    working_path = os.path.abspath('.')
    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)


    PADDING_SIZE = SETTINGS['padding size']
    VOCAB_SIZE = SETTINGS['vocab size']

    INPUT_DIM = VOCAB_SIZE
    OUTPUT_DIM = VOCAB_SIZE
    print('The input and output dimensions are:', INPUT_DIM, OUTPUT_DIM)
    ENC_EMB_DIM = 300
    HID_DIM = 300
    N_LAYERS = 1


    pkl_filename = os.path.join(working_path, 'final-ood-detector/Tokenizer_clinc_univsersal.pkl')
    # Load from file
    with open(pkl_filename, 'rb') as file:
        tokenizer = pickle.load(file)

    embd_matrix_savepath = os.path.join(working_path, 'final-ood-detector/Embd_matrix_clinc_universal.npy')
    embedding_matrix = np.load(embd_matrix_savepath)


    data_path = ''
    for title in ['ind', 'unlabeled']:
        if title == 'ind':
            data_path = os.path.join(working_path, 'data/finisheddata/ind_train_content_pad_finished.npy')
        elif title == 'unlabeled':
            data_path = os.path.join(working_path, 'data/finisheddata/unlabeled_content_pad_finished.npy')


        print('Now we are encoding ', title, 'text data into latent code...')
        padded_texts = np.load(data_path)


        seqs= tokenizer.texts_to_sequences(padded_texts)
        X = sequence.pad_sequences(seqs, maxlen=PADDING_SIZE, padding='post', truncating='post')


        train_X = X[:]
        x_tensor = torch.tensor(train_X, dtype=torch.long)

        ENCODER = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, embedding_matrix, N_LAYERS)

        path_state_dict = os.path.join(working_path, 'trainedmodels/ENC_params_clinc_ind_unlabeled.pth')
        state_dict_load = torch.load(path_state_dict)
        ENCODER.load_state_dict(state_dict_load)
        ENCODER.eval()

        src = x_tensor.permute(1,0)
        hidden, cell = ENCODER(src)
        # print(hidden.size())
        # print(cell.size())

        hidden = hidden.squeeze()
        cell = cell.squeeze()

        hidden = hidden.data.numpy()
        cell = cell.data.numpy()
        z = np.hstack((hidden, cell))
        # print(z.shape)

        save_path = ''
        if title == 'ind':
            save_path = os.path.join(working_path, 'data/latentcodedata/ind_train_latent_code_z.npy')
        elif title == 'unlabeled':
            save_path = os.path.join(working_path, 'data/latentcodedata/unlabeled_latent_code_z.npy')
        np.save(save_path,z)
        print('Save', title ,'latent code complete!!')

    print('Encoding text to latent code all complete!')
    print('\n')

if __name__ == '__main__':
    encode_data_to_vector()
