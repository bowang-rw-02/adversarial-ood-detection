import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from keras_preprocessing import sequence
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve
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
    def __init__(self, weights_matrix, class_num, kernel_num, kernel_sizes, dropout = 0.2):
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

def check_result_on_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    working_path = os.path.abspath('.')
    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)

    # static hyperparameters
    PADDING_SIZE = SETTINGS['padding size']
    CLASS_NUM = SETTINGS['label num']

    # custom hyperparameters
    KERNEL_NUM = 16
    KERNEL_SIZES = [3, 4, 5]


    pkl_filename = os.path.join(working_path, 'final-ood-detector/Tokenizer_clinc_univsersal.pkl')
    # Load from file
    with open(pkl_filename, 'rb') as file:
        tokenizer = pickle.load(file)

    embd_matrix_savepath = os.path.join(working_path, 'final-ood-detector/Embd_matrix_clinc_universal.npy')
    embedding_matrix = np.load(embd_matrix_savepath)

    test_texts_savepath = os.path.join(working_path, 'data/finisheddata/testset_content_pad_finished.npy')
    test_texts = np.load(test_texts_savepath)
    print('Loaded ', test_texts.shape, 'Test records.')

    seq = tokenizer.texts_to_sequences(test_texts)
    X = sequence.pad_sequences(seq, maxlen=PADDING_SIZE, padding='post', truncating='post')
    x_input = torch.tensor(X, dtype=torch.long)

    # because ind question has higher probability, so make the ind question as positive 1 label
    test_labels_savepath = os.path.join(working_path, 'data/finisheddata/testset_label.npy')
    Y = np.load(test_labels_savepath)

    MODES = ['INDC', 'OODD']


    for mode in MODES:
        TEXTCNN = TextCNN(embedding_matrix, CLASS_NUM, KERNEL_NUM, KERNEL_SIZES).to(device)
        if mode == 'INDC':
            path_state_dict = os.path.join(working_path, 'trainedmodels/IND_classifier_TextCNN_params.pth')
            complete_name = 'normal IND classifier'
        else:
            path_state_dict = os.path.join(working_path, 'final-ood-detector/OOD_detector_TextCNN_params.pth')
            complete_name = 'trained OOD detector'
        state_dict_load = torch.load(path_state_dict)
        TEXTCNN.load_state_dict(state_dict_load)
        TEXTCNN.eval()

        x_input = x_input.to(device)
        output = TEXTCNN(x_input)
        output = output.squeeze()
        output_P = F.softmax(output, dim=1)

        pred_y = torch.max(output_P, 1)
        pred_y_P = pred_y[0].detach().cpu().numpy()
        # pred_y_Label = pred_y[1].detach().numpy()
        pred_y_P = np.around(pred_y_P, decimals=8)

        precision, recall, thresholds = precision_recall_curve(Y, pred_y_P)
        PR_auc = auc(recall, precision)
        # print(thresholds.shape)

        print('For the ', complete_name, ' model, the auPR is: ', PR_auc, '.')



        fpr, tpr, thresholds = roc_curve(Y, pred_y_P)
        roc_auc = auc(fpr, tpr)
        print('For the ', complete_name, ' model, the auroc is: ', roc_auc, '.')
        # print(fpr, tpr)


        fpr95=1
        fpr90=1
        for ffpr,ttpr in zip(fpr,tpr):
            if abs(ttpr-0.95)<0.01:
                fpr95=ffpr
                break
        for ffpr,ttpr in zip(fpr,tpr):
            if abs(ttpr-0.90)<0.01:
                fpr90=ffpr
                break
        print('fpr95: ', fpr95, ' . fpr90: ', fpr90)


        if mode == 'OODD':
            # Calculate the G-mean
            gmean = np.sqrt(tpr * (1 - fpr))

            # Find the optimal threshold
            index = np.argmax(gmean)
            thresholdOpt = round(thresholds[index], ndigits=4)
            gmeanOpt = round(gmean[index], ndigits=4)
            fprOpt = round(fpr[index], ndigits=4)
            tprOpt = round(tpr[index], ndigits=4)

            print('OOD detector test on test data finished, we calculated and suggest you to use')
            print('The best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
            print('when utilizing the OOD detector.')
            print('With this threshold, the FPR: {}, TPR: {}'.format(fprOpt, tprOpt))
            print('We will also write it to exp_settings.json and threshold.json in the same folder.')

            SETTINGS['auroc']: str(roc_auc)
            SETTINGS['aupr']: str(PR_auc)
            SETTINGS['fpr95']: str(fpr95)
            SETTINGS['fpr90']: str(fpr90)
            SETTINGS['best threshold'] = str(thresholdOpt)
            with open(json_path, 'w') as f:
                json.dump(SETTINGS, f)

            th_json_path = os.path.join(working_path, 'final-ood-detector/exp_settings.json')
            with open(th_json_path, 'w') as f:
                json.dump(SETTINGS, f)

            print('Writing complete. ')


if __name__ == '__main__':
    check_result_on_test()




