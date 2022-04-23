import MeCab
import numpy as np
import os
import json
import sys
import nltk



def truncate_decide_cut():
    print('Calculating the propoer truncate/padding sizes ... ')
    working_path = os.path.abspath('.')
    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)

    # default truncation size
    TRUNCATE_SIZE = 30
    # default padding size = truncation size + sos + eos symbol
    PADDING_SIZE = TRUNCATE_SIZE + 2


    for title in ['unlabeled', 'ind', 'test']:
        data_path = ''
        if title == 'unlabeled':
            data_path = os.path.join(working_path, 'data/rawdata-cleaned/unlabeled_content_raw_cleaned.npy')
        elif title == 'ind':
            data_path = os.path.join(working_path, 'data/rawdata-cleaned/ind_train_content_raw_cleaned.npy')
        else:
            data_path = os.path.join(working_path, 'data/rawdata-cleaned/testset_content_raw_cleaned.npy')

        print('Now we are working on ', title, 'data.')


        orig_texts = np.load(data_path)

        # cut_sents = []
        cut_sents_lists = []

        for sent in orig_texts:
            cut_sent_list = nltk.word_tokenize(sent)
            # cut_sents.append(cut_sent)
            cut_sents_lists.append(cut_sent_list)


        if title == 'unlabeled':
            print('Calculating proper padding size using unlabeled data...')
            len_of_seq = np.zeros((61,), dtype=np.int)
            for j, eachseq in enumerate(cut_sents_lists):
                if len(eachseq) <= 60:
                    len_of_seq[len(eachseq)] = len_of_seq[len(eachseq)] + 1
                else:
                    len_of_seq[60] += 1
            # print(len_of_seq)

            good_trunc_size = 60
            good_data_cover_num = int(orig_texts.shape[0]*0.95)
            temp_count = 0

            print('The padding size should cover at least',good_data_cover_num,' (95\%) records.')
            for k in range(61):
                temp_count += len_of_seq[k]
                if temp_count >= good_data_cover_num:
                    if k<60 and len_of_seq[k+1]>100:
                        continue
                    else:
                        good_trunc_size = k
                        break

            # print('The padded size is decided as (',good_trunc_size,'+2).')

            TRUNCATE_SIZE = good_trunc_size
            PADDING_SIZE = TRUNCATE_SIZE + 2
            print('After the calculation on unlabeled data, we finally used ',PADDING_SIZE,'padding size.')


        cut_sents_SE= []
        for sent_list in cut_sents_lists:
            truncate_sent_list = sent_list[:TRUNCATE_SIZE]
            truncate_sent = ''
            for word in truncate_sent_list:
                truncate_sent += word
                truncate_sent += ' '
            cut_sent_SE = '<sos> ' + truncate_sent + '<eos>'
            cut_sents_SE.append(cut_sent_SE)

        print('Here are the first 10 cut sentences: ')
        # print(cut_sents_SE.shape)
        print(cut_sents_SE[:10])

        save_path_c = ''
        if title == 'unlabeled':
            save_path_c = os.path.join(working_path, 'data/finisheddata/unlabeled_content_pad_finished.npy')
        elif title == 'ind':
            save_path_c = os.path.join(working_path, 'data/finisheddata/ind_train_content_pad_finished.npy')
        else:
            save_path_c = os.path.join(working_path, 'data/finisheddata/testset_content_pad_finished.npy')
        np.save(save_path_c, cut_sents_SE)

        SETTINGS['padding size'] = PADDING_SIZE
        with open(json_path, 'w') as f:
            json.dump(SETTINGS, f)

    print('Sentence truncation complete.')
    print('\n')

if __name__ == '__main__':
    truncate_decide_cut()


