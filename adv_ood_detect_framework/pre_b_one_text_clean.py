import re
from string import punctuation
import numpy as np
import os


def text_clean():
    # basic clean version
    punc = punctuation + u'1234567890―/'
    print('We are going to exclude these symbols (stopwords) in text: \n', punc)

    working_path = os.path.abspath('.')
    for i in ['ind_train', 'ood_train', 'ind_test', 'ood_test']:
        data_path = os.path.join(working_path, 'data/rawdata/' + i + '_content_raw.npy')

        fr = np.load(data_path)
        fw = []


        for line in fr:
            nline = re.sub(r"[%s]+" %punc,"", line)
            fw.append(nline)

        fw = np.array(fw)
        # print(fw[:10])
        # print(fw.shape)

        save_path_c = os.path.join(working_path, ('data/rawdata-cleaned/' + i + '_content_raw_cleaned.npy'))
        # print(save_path_c)

        np.save(save_path_c, fw)
        print(i, 'text data cleaning complete.')

    print('Text cleaning all completed.')
    print('\n')

if __name__ == '__main__':
    text_clean()