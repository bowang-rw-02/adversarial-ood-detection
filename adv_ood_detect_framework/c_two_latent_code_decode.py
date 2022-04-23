import torch
import torch.nn as nn
import numpy as np
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


def decode_generate_code():
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


    ind_data_path = os.path.join(working_path, 'data/finisheddata/ind_train_content_pad_finished.npy')
    ind_data = np.load(ind_data_path)
    IND_DATA_NUM = ind_data.shape[0]

    GAN_DIM = ENC_EMB_DIM*2
    generator = Generator(GAN_DIM,GAN_DIM,GAN_DIM)

    path_state_dict = os.path.join(working_path, 'trainedmodels/GAN_GEN_params_ind_unlabeled.pth')
    state_dict_load = torch.load(path_state_dict)
    generator.load_state_dict(state_dict_load)
    generator.eval()

    DECODER = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS)
    path_state_dict = os.path.join(working_path, 'trainedmodels/DEC_params_clinc_ind_unlabeled.pth')
    state_dict_load = torch.load(path_state_dict)
    DECODER.load_state_dict(state_dict_load)
    DECODER.eval()

    FAKE_OOD_TIMES = 1
    rBATCH_SIZE = IND_DATA_NUM * FAKE_OOD_TIMES
    print('We are going to generate ', rBATCH_SIZE, 'fake oods,',FAKE_OOD_TIMES, 'time(s) to IND data num.')


    noise = torch.FloatTensor(rBATCH_SIZE, GAN_DIM)
    torch.randn((rBATCH_SIZE, GAN_DIM), out=noise)

    generated_data = generator(noise)


    # [rBATCH_SIZE,600]
    split_g_data = torch.split(generated_data, 300 , dim=1)
    hidden = split_g_data[0].unsqueeze(0)
    cell = split_g_data[1].unsqueeze(0)

    # when the tokenizer has 'oov' symbol, the id of <sos> is 2 not 1
    START_TOKEN_ID = 2
    input = torch.ones(rBATCH_SIZE)*START_TOKEN_ID
    input = input.long()
    outputs = torch.zeros(PADDING_SIZE, rBATCH_SIZE, OUTPUT_DIM)


    for t in range(1, PADDING_SIZE):
        output, hidden, cell = DECODER(input, hidden, cell)
        outputs[t] = output

        # decoder output: [output dim, batch size]

        top1 = output.argmax(1)
        input = top1


    # print(outputs.cpu().shape)
    output_result = outputs.cpu().permute(1,0,2).argmax(2)
    # if the tokenizer contains word 'oov', the id of 'sos' is 2 rather than 1
    output_result[:,0] = START_TOKEN_ID
    output_result_np = output_result.detach().numpy()
    # print(output_result_np.shape)

    fake_ood_savepath = os.path.join(working_path, ('data/generateddata/fake_ood_' +str(FAKE_OOD_TIMES)+ 'times_to_ind_size.npy'))
    np.save(fake_ood_savepath, output_result_np)
    print('Save fake OOD data complete!')
    # print(output_result_np[:10])
    print('\n')


if __name__ == '__main__':
    decode_generate_code()


