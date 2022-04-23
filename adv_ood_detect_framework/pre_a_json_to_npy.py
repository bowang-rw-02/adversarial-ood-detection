import json
import numpy as np
import os


def read_json_to_npy():
    working_path = os.path.abspath('.')
    for i in ['ind_train', 'ood_train', 'ind_test', 'ood_test']:
        data_path = ''
        data_path2 = ''
        if i == 'ind_train':
            data_path = os.path.join(working_path, 'oridata/ind_train.json')
            data_path2 = os.path.join(working_path, 'oridata/ind_val.json')
        elif i == 'ood_train':
            data_path = os.path.join(working_path, 'oridata/ood_train_plus.json')
            data_path2 = os.path.join(working_path, 'oridata/ood_val.json')
        elif i == 'ind_test':
            data_path = os.path.join(working_path, 'oridata/ind_test.json')
        else:
            data_path = os.path.join(working_path, 'oridata/ood_test.json')

        f = open(data_path, 'r')
        data = json.load(f)
        f.close()
        if i == 'ind_train' or i == 'ood_train':
            f2 = open(data_path2, 'r')
            data2 = json.load(f2)
            f2.close()


        contents = []
        labels = []
        count = 0
        for line in data:
            contents.append(line[0])
            labels.append(line[1])
            count = count + 1
        if i == 'ind_train' or i == 'ood_train':
            for line2 in data2:
                contents.append(line2[0])
                labels.append(line2[1])
                count = count + 1

        contents = np.array(contents)
        labels = np.array(labels)

        save_path_c = ''
        save_path_l = ''
        if i == 'ind_train':
            save_path_c = os.path.join(working_path, 'data/rawdata/ind_train_content_raw.npy')
            save_path_l = os.path.join(working_path, 'data/rawdata/ind_train_label_raw.npy')
            np.save(save_path_c, contents)
            np.save(save_path_l, labels)
        else:
            save_path_c = os.path.join(working_path, ('data/rawdata/'+ i + '_content_raw.npy'))
            np.save(save_path_c, contents)

        print('Here are the first 10 records of ', i, ':' )
        print(contents[:10])
        print(labels[:10])
        print('Save ',i,' data complete. Writting ', count, ' records.')
        print('\n')

    print('\n')

if __name__ == '__main__':
    read_json_to_npy()