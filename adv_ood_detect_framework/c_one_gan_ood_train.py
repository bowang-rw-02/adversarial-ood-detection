import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import json

class Generator(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.hidden1 = nn.Linear(input_dim, hid_dim)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.ReLU()
        self.hidden3 = nn.Linear(hid_dim, out_dim)


    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = torch.tanh(self.hidden3(x))

        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super().__init__()

        self.hidden1 = nn.Linear(input_dim, hid_dim)
        self.relu1 = nn.LeakyReLU(0.2)
        self.hidden2 = nn.Linear(hid_dim, hid_dim)
        self.relu2 = nn.LeakyReLU(0.2)
        self.predict = nn.Linear(hid_dim, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu1(x)
        x = self.hidden2(x)
        x = self.relu2(x)
        x = self.predict(x)
        x = self.sigmoid(x)

        return x

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


def gan_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    working_path = os.path.abspath('.')
    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)

    INPUT_DIM = 600
    OUT_DIM = 600
    HID_DIM = 600
    BATCH_SIZE = 128
    BATCH_SIZE_UN = 128
    LR = 0.0002
    N_EPOCHS = 200

    PADDING_SIZE = SETTINGS['padding size']
    VOCAB_SIZE = SETTINGS['vocab size']
    De_OUTPUT_DIM = VOCAB_SIZE
    De_DEC_EMB_DIM = 300
    De_HID_DIM = 300
    De_N_LAYERS = 1


    DECODER = Decoder(De_OUTPUT_DIM, De_DEC_EMB_DIM, De_HID_DIM, De_N_LAYERS).to(device)
    path_state_dict = os.path.join(working_path, 'trainedmodels/DEC_params_clinc_ind_unlabeled.pth')
    state_dict_load = torch.load(path_state_dict)
    DECODER.load_state_dict(state_dict_load)
    DECODER.eval()



    KERNEL_NUM = 16
    KERNEL_SIZES = [3, 4, 5]
    CLASS_NUM = SETTINGS['label num']

    embd_matrix_savepath = os.path.join(working_path, 'final-ood-detector/Embd_matrix_clinc_universal.npy')
    embedding_matrix = np.load(embd_matrix_savepath)
    TEXTCNN = TextCNN(embedding_matrix, CLASS_NUM, KERNEL_NUM, KERNEL_SIZES).to(device)
    path_state_dict = os.path.join(working_path, 'trainedmodels/IND_classifier_TextCNN_params.pth')
    state_dict_load = torch.load(path_state_dict)
    TEXTCNN.load_state_dict(state_dict_load)
    TEXTCNN.eval()

    generator = Generator(INPUT_DIM, HID_DIM, OUT_DIM).to(device)
    discriminator = Discriminator(OUT_DIM, HID_DIM).to(device)


    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)
    loss_func = nn.BCELoss()

    true_data_path = os.path.join(working_path, 'data/latentcodedata/ind_train_latent_code_z.npy')
    true_data = np.load(true_data_path)
    true_data_tensor = torch.tensor(true_data, dtype=torch.float)
    true_data_loader = DataLoader(true_data_tensor, batch_size=BATCH_SIZE, shuffle=True)

    true_data_un_path = os.path.join(working_path, 'data/latentcodedata/unlabeled_latent_code_z.npy')
    true_data_un = np.load(true_data_un_path)
    true_data_tensor_un = torch.tensor(true_data_un, dtype=torch.float)
    true_data_loader_un = DataLoader(true_data_tensor_un, batch_size=BATCH_SIZE_UN, shuffle=True)

    ITER_TIME = int(true_data.shape[0]/BATCH_SIZE)
    BATCH_SIZE_UN = int(true_data_un.shape[0]/ITER_TIME)
    print('The batch size of unlabeled latent code is reset to: ', BATCH_SIZE_UN)


    G_losses = []
    D_losses = []
    D_losses_mix = []
    iters = 0

    for epoch in range(N_EPOCHS):
        generator.train()
        discriminator.train()

        for step, (true_data_batch, true_data_batch_un) in enumerate(zip(true_data_loader, true_data_loader_un)):
            rBATCH_SIZE = true_data_batch.shape[0]
            true_data_batch = true_data_batch.to(device)

            noise = torch.cuda.FloatTensor(rBATCH_SIZE, INPUT_DIM)
            torch.randn((rBATCH_SIZE, INPUT_DIM), out=noise)
            true_labels = torch.cuda.FloatTensor(rBATCH_SIZE, 1).fill_(1)
            fake_labels = torch.cuda.FloatTensor(rBATCH_SIZE, 1).fill_(0)

            generated_data = generator(noise)

            # TRAIN DISCRIMINATOR
            # we train the discriminator to identify (remember) true IND data
            # if D_G_z1>=0.5 or D_G_z2>=0.5:
            discriminator_optimizer.zero_grad()
            true_discriminator_out = discriminator(true_data_batch)
            true_discriminator_loss = loss_func(true_discriminator_out, true_labels)
            true_discriminator_loss.backward()
            D_x = true_discriminator_out.unsqueeze(1).mean().item()

            # we cannot let genetator be trained by the gradient link of generated_data so add detach
            generator_discriminator_out = discriminator(generated_data.detach())
            generator_discriminator_loss = loss_func(generator_discriminator_out, fake_labels)
            generator_discriminator_loss.backward()
            D_G_z1 = generator_discriminator_out.unsqueeze(1).mean().item()


            discriminator_loss = true_discriminator_loss + generator_discriminator_loss
            discriminator_optimizer.step()

            ###
            rBATCH_SIZE_un = true_data_batch_un.shape[0]
            true_data_batch_un = true_data_batch_un.to(device)

            noise_un = torch.cuda.FloatTensor(rBATCH_SIZE_un, INPUT_DIM)
            torch.randn((rBATCH_SIZE_un, INPUT_DIM), out=noise_un)
            true_labels_un = torch.cuda.FloatTensor(rBATCH_SIZE_un, 1).fill_(1)
            fake_labels_un = torch.cuda.FloatTensor(rBATCH_SIZE_un, 1).fill_(0)

            generated_data_un = generator(noise_un)

            # TRAIN DISCRIMINATOR with unlabeled data
            # we train the discriminator to identify (remember) true unlabeled data
            # if D_G_z1>=0.5 or D_G_z2>=0.5:
            discriminator_optimizer.zero_grad()
            true_discriminator_out_un = discriminator(true_data_batch_un)
            true_discriminator_loss_un = loss_func(true_discriminator_out_un, true_labels_un)
            true_discriminator_loss_un.backward()
            D_x_mix = true_discriminator_out_un.unsqueeze(1).mean().item()

            # we cannot let genetator be trained by the gradient link of generated_data so add detach
            generator_discriminator_out_un = discriminator(generated_data_un.detach())
            generator_discriminator_loss_un = loss_func(generator_discriminator_out_un, fake_labels_un)
            generator_discriminator_loss_un.backward()
            D_G_z1_m = generator_discriminator_out_un.unsqueeze(1).mean().item()

            discriminator_loss_m = true_discriminator_loss_un + generator_discriminator_loss_un
            discriminator_optimizer.step()
            ###



            # TRAIN GENERATOR
            # train generator. The reason why we also use the 'true label' is
            # because we want generator to generate data that discriminator think is true
            generator_optimizer.zero_grad()

            generator_discriminator_out = discriminator(generated_data)
            generator_loss = loss_func(generator_discriminator_out, true_labels)

            D_G_z2 = generator_discriminator_out.unsqueeze(1).mean().item()

            # Here we train the generator again using AC's result
            hidden, cell = generated_data.split(300, dim=1)
            hidden = hidden.unsqueeze(0).contiguous()
            cell = cell.unsqueeze(0).contiguous()
            uniform_dist = torch.Tensor(rBATCH_SIZE, CLASS_NUM).fill_((1. / CLASS_NUM))
            uniform_dist = uniform_dist.to(device)

            START_TOKEN_ID = 2
            # id of token <sos> (the initial input of encoder)
            # if you defined the unk (oov) token when constructing tokenizer,
            # be careful to remember that the id of oov is 1, sos is 2! If you use bert tokenizer, sos 101 eos 102 pad 0
            input = torch.cuda.LongTensor(rBATCH_SIZE).fill_(START_TOKEN_ID)
            outputs = torch.cuda.FloatTensor(PADDING_SIZE, rBATCH_SIZE, De_OUTPUT_DIM).fill_(0)
            for t in range(1, PADDING_SIZE):
                output, hidden, cell = DECODER(input, hidden, cell)
                outputs[t] = output
                # decoder output: [output dim, batch size]
                top1 = output.argmax(1)
                input = top1
            output_result = outputs.permute(1, 0, 2).argmax(2)
            output_result[:, 0] = START_TOKEN_ID
            # generator_classifier_out = classifier(generated_data_seq)
            generator_classifier_out = TEXTCNN(output_result)
            generator_classifier_out = generator_classifier_out.squeeze()
            # generator_classifier_out_P = F.softmax(generator_classifier_out, dim=1)
            # generator_AC_loss = (generator_classifier_out_P*torch.log2(generator_classifier_out_P+EPS)).sum()
            KL_fake_output = F.log_softmax(generator_classifier_out, dim=1)
            generator_AC_loss = F.kl_div(KL_fake_output, uniform_dist)*CLASS_NUM


            generator_total_loss = generator_loss+generator_AC_loss
            generator_total_loss.backward()
            generator_optimizer.step()


            if step == int(ITER_TIME/2) or step == ITER_TIME-1:
                # print(generator_classifier_out_P.shape)
                print('[%d/%d][%d/%d]\t\t\tLoss_D: %.4f\tLoss_G: %.4f\t%.4f\t%.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t For unlabeled data D(x): %.4f\tD(G(z)): %.4f'
                      % (epoch, N_EPOCHS, step, len(true_data_loader),
                         discriminator_loss.item(), generator_loss.item(), generator_AC_loss.item(), generator_total_loss.item(), D_x, D_G_z1, D_G_z2, D_x_mix, D_G_z1_m))


            G_losses.append(generator_total_loss.item())
            D_losses.append(discriminator_loss.item())
            D_losses_mix.append(discriminator_loss_m.item())
            iters += 1


    gen_savepath = os.path.join(working_path, 'trainedmodels/GAN_GEN_params_ind_unlabeled.pth')
    torch.save(generator.state_dict(), gen_savepath)
    dis_savepath = os.path.join(working_path, 'trainedmodels/GAN_DIS_params_ind_unlabeled.pth')
    torch.save(discriminator.state_dict(), dis_savepath)
    print('Save GAN GEN DIS model complete.')
    print('\n')


if __name__ == '__main__':
    gan_training()

