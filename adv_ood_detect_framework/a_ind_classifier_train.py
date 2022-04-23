import os
import numpy as np
from keras_preprocessing import sequence
# from keras.preprocessing import sequence
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import json


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    # how many words in dict (matrix), embedding dim
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    emb_layer.weight = nn.Parameter(torch.tensor(weights_matrix, dtype=torch.float32))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class TextCNN(nn.Module):
    def __init__(self, weights_matrix, class_num, kernel_num, kernel_sizes):
        super(TextCNN, self).__init__()
        Ci = 1
        Co = kernel_num

        self.embd, embd_num, embd_dim = create_emb_layer(weights_matrix, False)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (k, embd_dim)) for k in kernel_sizes])
        # self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(Co * len(kernel_sizes), class_num)

    def forward(self, x):
        x = self.embd(x) # (batch_N, token_num(word in one sent), embd_dim)
        x = x.unsqueeze(1) # (N, Ci(channel, for text only 1, token_num, embd_dim)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1) # concat results of 3 size kernel
        # x = self.dropout(x)
        logit = self.fc(x)
        return logit




def train_ind_classifier():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    working_path = os.path.abspath('.')
    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)

    # static hyperparameters
    PADDING_SIZE = SETTINGS['padding size']
    CLASS_NUM = SETTINGS['label num']

    # custom hyperparameters
    LR = 0.001
    EPOCH = 7
    BATCH_SIZE = 64
    EMBD_DIM = 300
    KERNEL_NUM = 16
    KERNEL_SIZES = [3, 4, 5]




    pkl_filename = os.path.join(working_path, 'final-ood-detector/Tokenizer_clinc_univsersal.pkl')
    # Load from file
    with open(pkl_filename, 'rb') as file:
        tokenizer = pickle.load(file)

    embd_matrix_savepath = os.path.join(working_path, 'final-ood-detector/Embd_matrix_clinc_universal.npy')
    embedding_matrix = np.load(embd_matrix_savepath)

    ind_texts_savepath = os.path.join(working_path, 'data/finisheddata/ind_train_content_pad_finished.npy')
    padded_ind_texts = np.load(ind_texts_savepath)
    seq = tokenizer.texts_to_sequences(padded_ind_texts)
    X = sequence.pad_sequences(seq, maxlen=PADDING_SIZE, padding='post', truncating='post')
    # print(X[:10])
    ind_labels_savepath = os.path.join(working_path, 'data/finisheddata/ind_train_label_numbered.npy')
    y = np.load(ind_labels_savepath)

    train_X = X[:]
    train_y = y[:]

    # Process data as tensor
    x_train = torch.tensor(train_X, dtype=torch.long)
    y_train = torch.tensor(train_y, dtype=torch.long)
    # print(y_train[:10])



    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    model = TextCNN(embedding_matrix, CLASS_NUM, KERNEL_NUM, KERNEL_SIZES).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()

    iterations = int(padded_ind_texts.shape[0]/BATCH_SIZE)
    for epoch in range(EPOCH):
        model.train()
        for step, (x_batch, y_batch) in enumerate(train_loader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            output = output.squeeze()
            loss = loss_func(output, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == iterations-1:
                output = output.cpu()
                y_batch = y_batch.cpu()
                train_pred_y = torch.max(output, 1)[1].data.numpy()

                train_target_y = y_batch.data.numpy()
                train_accuracy = float((train_pred_y == train_target_y).astype(int).sum())/float(train_target_y.size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.4f' % train_accuracy)

    model_savepath = os.path.join(working_path, 'trainedmodels/IND_classifier_TextCNN_params.pth')
    torch.save(model.state_dict(), model_savepath)
    print('save IND classifier model complete!')

if __name__ == '__main__':
    train_ind_classifier()