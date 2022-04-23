import numpy as np
from keras_preprocessing.text import Tokenizer
from keras_preprocessing import sequence
import fasttext
import fasttext.util
import pickle
import os
import json


def tokenizer_matrix_build():
    working_path = os.path.abspath('.')
    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)
    PADDING_SIZE = SETTINGS['padding size']

    padded_sent_path1 = os.path.join(working_path, 'data/finisheddata/ind_train_content_pad_finished.npy')
    padded_sent_path2 = os.path.join(working_path, 'data/finisheddata/unlabeled_content_pad_finished.npy')
    padded_sent_path3 = os.path.join(working_path, 'data/finisheddata/testset_content_pad_finished.npy')

    padded_sents1 = np.load(padded_sent_path1)
    padded_sents2 = np.load(padded_sent_path2)
    padded_sents3 = np.load(padded_sent_path3)

    padded_sents_all = np.concatenate((padded_sents1,padded_sents2,padded_sents3), axis=0)
    padded_sents_all_len = padded_sents_all.shape[0]

    o_tokenizer = Tokenizer(oov_token='oov')
    # o_tokenizer = Tokenizer()
    o_tokenizer.fit_on_texts(padded_sents_all)
    # print(o_tokenizer.word_index)
    # {'oov': 1, 'sos': 2, 'eos': 3, ...

    print('The total words num are: ', len(o_tokenizer.word_index)+1)
    # print(o_tokenizer.word_counts)
    # print(sorted(o_tokenizer.word_counts.items(), key=lambda item:item[1], reverse=True))


    # Save tokenizer file in the current working directory
    pkl_filename = os.path.join(working_path, 'final-ood-detector/Tokenizer_clinc_univsersal.pkl')
    with open(pkl_filename, 'wb') as file:
        pickle.dump(o_tokenizer, file)


    vocab_size = len(o_tokenizer.word_index)+1
    fasttext.util.download_model('en', if_exists='ignore')
    ft = fasttext.load_model('cc.en.300.bin')

    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in o_tokenizer.word_index.items():
        embedding_vector = ft.get_word_vector(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    embd_matrix_savepath = os.path.join(working_path, 'final-ood-detector/Embd_matrix_clinc_universal.npy')
    np.save(embd_matrix_savepath, embedding_matrix)
    print('The shape of embedding matrix is: ', embedding_matrix.shape)

    SETTINGS['vocab size'] = vocab_size
    with open(json_path, 'w') as f:
        json.dump(SETTINGS, f)

    print('Tokenizer and embedding matrix preparation complete.')
    print('\n')


if __name__ == '__main__':
    tokenizer_matrix_build()






