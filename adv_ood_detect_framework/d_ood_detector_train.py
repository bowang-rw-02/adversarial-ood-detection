import numpy as np
from keras_preprocessing import sequence
# from keras.preprocessing import sequence
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import os
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

        # self.embd_dim = embd_dim
        # self.embd = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        # self.embd.weight = nn.Parameter(torch.tensor(weights_matrix, dtype=torch.float32))
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


def train_ood_detector():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    working_path = os.path.abspath('.')
    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)

    # static hyperparameters
    PADDING_SIZE = SETTINGS['padding size']
    CLASS_NUM = SETTINGS['label num']

    # custom hyperparameters
    LR = 0.0002
    EPOCH = 19
    BATCH_SIZE = 64
    KERNEL_NUM = 16
    KERNEL_SIZES = [3, 4, 5]


    pkl_filename = os.path.join(working_path, 'final-ood-detector/Tokenizer_clinc_univsersal.pkl')
    # Load from file
    with open(pkl_filename, 'rb') as file:
        tokenizer = pickle.load(file)

    embd_matrix_savepath = os.path.join(working_path, 'final-ood-detector/Embd_matrix_clinc_universal.npy')
    embedding_matrix = np.load(embd_matrix_savepath)

    ind_texts_savepath = os.path.join(working_path, 'data/finisheddata/ind_train_content_pad_finished.npy')
    ind_texts = np.load(ind_texts_savepath)

    fake_ood_tokens_savepath = os.path.join(working_path, 'data/generateddata/fake_ood_1times_to_ind_size.npy')
    X_OOD = np.load(fake_ood_tokens_savepath)
    x_ood_train = torch.tensor(X_OOD, dtype=torch.long)


    seq = tokenizer.texts_to_sequences(ind_texts)
    X = sequence.pad_sequences(seq, maxlen=PADDING_SIZE, padding='post', truncating='post')
    ind_labels_savepath = os.path.join(working_path, 'data/finisheddata/ind_train_label_numbered.npy')
    y = np.load(ind_labels_savepath)

    train_X = X[:]
    train_y = y[:]


    # Process data as tensor
    x_train = torch.tensor(train_X, dtype=torch.long)
    y_train = torch.tensor(train_y, dtype=torch.long)


    train = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_oe = DataLoader(x_ood_train, batch_size=int(BATCH_SIZE*1), shuffle=True)

    model = TextCNN(embedding_matrix, CLASS_NUM, KERNEL_NUM, KERNEL_SIZES).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    iterations = int(ind_texts.shape[0] / BATCH_SIZE)
    for epoch in range(EPOCH):
        model.train()
        for step, ((x_batch, y_batch), x_ood_batch) in enumerate(zip(train_loader, train_loader_oe)):
            # print(step)
            uniform_dist = torch.Tensor(x_ood_batch.size(0), CLASS_NUM).fill_((1. / CLASS_NUM))
            uniform_dist = uniform_dist.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x_ood_batch = x_ood_batch.to(device)


            output = model(x_batch)
            output = output.squeeze()
            loss_ind = F.cross_entropy(output, y_batch)


            output_ood = model(x_ood_batch)
            output_ood = output_ood.squeeze()
            KL_fake_output = F.log_softmax(output_ood, dim=1)
            # when you use kl_div(q.log(),p) you need to do log for q first, that's why we use log_softmax to output
            loss_ood = F.kl_div(KL_fake_output, uniform_dist)*CLASS_NUM



            loss = loss_ind + loss_ood
            # loss = loss_ind

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            if step == iterations-1:
                output = output.cpu()
                y_batch = y_batch.cpu()
                train_pred_y = torch.max(output, 1)[1].data.numpy()
                train_target_y = y_batch.data.numpy()
                train_accuracy = float((train_pred_y == train_target_y).astype(int).sum())/float(train_target_y.size)
                print('Epoch: ', epoch, '| ind loss: %.4f' % loss_ind.cpu().data.numpy(), '| ood loss: %.4f' % loss_ood.cpu().data.numpy(), '| total loss: %.4f' % loss.cpu().data.numpy(), '| accuracy: %.4f' % train_accuracy)

    model_savepath = os.path.join(working_path, 'final-ood-detector/OOD_detector_TextCNN_params.pth')
    torch.save(model.state_dict(), model_savepath)
    print('save IND classifier model complete!')


if __name__ == '__main__':
    train_ood_detector()
