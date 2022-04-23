import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import DataLoader
import numpy as np
from keras_preprocessing import sequence
import pickle
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
        # cell = similar to hidden

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc_h(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        hidden = hidden.unsqueeze(0)
        # hidden = [batch size, hid dim]
        cell = torch.tanh(self.fc_c(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim = 1)))
        cell = cell.unsqueeze(0)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.LSTM = nn.LSTM(input_size= emb_dim,
                            hidden_size= hid_dim,
                            num_layers= n_layers,
                            bidirectional= False,
                            batch_first= False)

        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden, cell):
        # [word 1,2,3,4] to [[word 1],[2],[3],[4]]
        input = input.unsqueeze(0)

        embdded = self.embedding(input)
        # embdded = [1, batch size, emb dim]
        output, (hidden, cell) = self.LSTM(embdded, (hidden, cell))
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, iter_time, total_epoch):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.iter_time = iter_time
        self.total_epoch = total_epoch
        self.iteration = 0

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # trg = [trg sent len, batch]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # the <sos> is used as input
        input = trg[0, :]

        self.iteration += 1
        if self.iteration%self.iter_time==0:
            # print('The iteration time is:')
            # print(self.iteration)
            cut_value = self.iteration//self.iter_time
            # reduce the value of teacher forcing as the training proceeds
            teacher_forcing_ratio = 1-np.sqrt(cut_value/self.total_epoch)
            print('now teacher foring value is:')
            print(teacher_forcing_ratio)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            # because decoder is batch first...
            outputs[t] = output
            # print('before we get max', output.size())
            teacher_force = random.random() < teacher_forcing_ratio
            # decoder output: [batch size, output dim]
            top1 = output.argmax(1)
            # print('after we get max', top1.size())
            input = trg[t] if teacher_force else top1
        return outputs



def train_autoencoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    DEC_EMB_DIM = 300
    HID_DIM = 300
    N_LAYERS = 1
    N_EPOCHS = 45
    BATCH_SIZE = 128
    LR = 0.0003




    pkl_filename = os.path.join(working_path, 'final-ood-detector/Tokenizer_clinc_univsersal.pkl')
    # Load from file
    with open(pkl_filename, 'rb') as file:
        tokenizer = pickle.load(file)

    embd_matrix_savepath = os.path.join(working_path, 'final-ood-detector/Embd_matrix_clinc_universal.npy')
    embedding_matrix = np.load(embd_matrix_savepath)


    ind_texts_savepath = os.path.join(working_path, 'data/finisheddata/ind_train_content_pad_finished.npy')
    ind_texts_padded = np.load(ind_texts_savepath)
    unlabeled_texts_savepath = os.path.join(working_path, 'data/finisheddata/unlabeled_content_pad_finished.npy')
    unlabeled_texts_padded = np.load(unlabeled_texts_savepath)


    seqs = tokenizer.texts_to_sequences(ind_texts_padded)
    seqs_un = tokenizer.texts_to_sequences(unlabeled_texts_padded)

    X = sequence.pad_sequences(seqs, maxlen=PADDING_SIZE, padding='post', truncating='post')
    Y = sequence.pad_sequences(seqs, maxlen=PADDING_SIZE, padding='post', truncating='post')

    X_un = sequence.pad_sequences(seqs_un, maxlen=PADDING_SIZE, padding='post', truncating='post')
    Y_un = sequence.pad_sequences(seqs_un, maxlen=PADDING_SIZE, padding='post', truncating='post')


    train_X = X[:]
    train_Y = Y[:]
    train_X_un = X_un[:]
    train_Y_un = Y_un[:]

    x_train = torch.tensor(train_X, dtype=torch.long)
    y_train = torch.tensor(train_Y, dtype=torch.long)
    x_train_m = torch.tensor(train_X_un, dtype=torch.long)
    y_train_m = torch.tensor(train_Y_un, dtype=torch.long)

    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    train_unlabeled = torch.utils.data.TensorDataset(x_train_m, y_train_m)
    train_loader_unlabeled = DataLoader(train_unlabeled, batch_size=BATCH_SIZE, shuffle=True)

    losses = []
    losses_un = []


    ITER_TIME = int((ind_texts_padded.shape[0] + unlabeled_texts_padded.shape[0]) / BATCH_SIZE)


    TRAINING_FLAG = 1
    while TRAINING_FLAG:
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, embedding_matrix, N_LAYERS)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)
        model = Seq2Seq(enc, dec, device, ITER_TIME, N_EPOCHS).to(device)

        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.08, 0.08)
        model.apply(init_weights)
        print(model)

        optimizer = optim.Adam(model.parameters(), lr=LR)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(N_EPOCHS):
            model.train()
            epoch_loss = 0
            epoch_loss_un = 0

            for step, (x_batch, y_batch) in enumerate(train_loader):
                # x_batch [batch size, sent len]
                # after permute [sent len, batch size]
                src = x_batch.permute(1,0).to(device)
                trg = y_batch.permute(1,0).to(device)


                optimizer.zero_grad()

                output = model(src, trg)
                # trg = [trg len, batch size]
                # output = [trg len, batch size, output_dim]
                output_dim = output.shape[-1]

                # output2 = output[1:].view(-1, output_dim)
                # trg2 = trg[1:].view(-1)
                output2 = output[1:].reshape(-1, output_dim)
                trg2 = trg[1:].reshape(-1)
                # reshape(-1) means the array is reshaped to an 1 dim array
                # output2 = [trg len * batchsize , ouput dim]
                # trg2 = [trg len * batchsize]

                loss = loss_func(output2, trg2)
                # as the loss function only works on 2d inputs with 1d targets
                # we need to flatten each of them with .view
                # we slice off the first column of the output and target tensors as mentioned above
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                epoch_loss+=loss.item()

            print('----------------------------------------------')
            print('For epoch ', epoch, ', the pure ind loss is:')
            epoch_loss_per = epoch_loss/len(train_loader)
            print(epoch_loss_per)
            losses.append(epoch_loss_per)


            for step, (x_batch_un, y_batch_un) in enumerate(train_loader_unlabeled):
                src = x_batch_un.permute(1,0).to(device)
                trg = y_batch_un.permute(1,0).to(device)

                optimizer.zero_grad()

                output = model(src, trg)
                output_dim = output.shape[-1]
                output2 = output[1:].reshape(-1, output_dim)
                trg2 = trg[1:].reshape(-1)
                loss = loss_func(output2, trg2)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                epoch_loss_un+=loss.item()


            print('For epoch ', epoch, ', the unlabeled loss is:')
            epoch_loss_per_un = epoch_loss_un/len(train_loader_unlabeled)
            print(epoch_loss_per_un)
            losses_un.append(epoch_loss_per_un)

            if epoch_loss_per<=0.2 and epoch_loss_per_un <= 0.2:
                print('Enough, we got a good en/decoder.')
                TRAINING_FLAG = 0
                break

            if epoch >= 30:
                if epoch_loss_per >= 0.5 and epoch_loss_per_un >= 0.5:
                    print('The training procedure of autoencoder this time is unstable, retraining from start...')
                    break

    model_savepath_e = os.path.join(working_path, 'trainedmodels/ENC_params_clinc_ind_unlabeled.pth')
    torch.save(enc.state_dict(), model_savepath_e)
    model_savepath_d = os.path.join(working_path, 'trainedmodels/DEC_params_clinc_ind_unlabeled.pth')
    torch.save(dec.state_dict(), model_savepath_d)
    print('Save Encoder Decoder model complete.')
    print('\n')

if __name__ == '__main__':
    train_autoencoder()