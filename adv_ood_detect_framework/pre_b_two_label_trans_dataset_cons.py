import numpy as np
import pickle
import os
import json

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def label_transform():
    print('Transforming ind text labels to numbers...')
    working_path = os.path.abspath('.')
    labels_path = os.path.join(working_path, 'data/rawdata/ind_train_label_raw.npy')
    labels = np.load(labels_path)

    label_dict = {}
    label_dict_r = {}
    count_label = 0

    for label in labels:
        if label not in label_dict:
            label_dict[label]=count_label
            label_dict_r[count_label]=label
            count_label+=1

    label_dict['oos']=count_label
    label_dict_r[count_label]='oos'

    # print(label_dict)
    # print(label_dict_r)

    numbered_labels = []
    for label in labels:
        numbered_labels.append(label_dict[label])
    numbered_labels = np.array(numbered_labels)
    # print(numbered_labels[:100])
    # print(numbered_labels.shape)


    dict_save_path = os.path.join(working_path, 'otherdatafiles/ind_train_label_text_num_dict.pkl')
    dictR_save_path = os.path.join(working_path, 'otherdatafiles/ind_train_label_num_text_dictR.pkl')
    # 也许可以加个日期
    save_dict(label_dict, dict_save_path)
    save_dict(label_dict_r, dictR_save_path)

    c_labels_save_path = os.path.join(working_path, 'data/finisheddata/ind_train_label_numbered.npy')
    np.save(c_labels_save_path, numbered_labels)

    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)
    SETTINGS['label num'] = count_label
    with open(json_path, 'w') as f:
        json.dump(SETTINGS, f)

    print('IND labels transformation complete, the real total category num is: ', count_label)
    print('\n')

def dataset_construct():
    working_path = os.path.abspath('.')

    json_path = os.path.join(working_path, 'exp_settings.json')
    with open(json_path, 'r') as f:
        SETTINGS = json.load(f)

    # static hyperparameters
    CLASS_NUM = SETTINGS['label num']


    data_dict = {}
    for i in ['ind_train', 'ood_train', 'ind_test', 'ood_test']:
        data_dict[i] = np.load(os.path.join(working_path, 'data/rawdata-cleaned/' + i + '_content_raw_cleaned.npy'))

    # construct the unlabeled data
    ind_data_used = data_dict['ind_train'][:15000]
    # print(ind_data_used.shape, ind_data_used[:10])
    ood_data_used = data_dict['ood_train']
    ind_ratio = 66
    unlabeled_data = np.array(ind_data_used[:ind_ratio])
    for i in range(1, CLASS_NUM):
        unlabeled_data = np.concatenate((unlabeled_data, ind_data_used[i*100:ind_ratio+i*100]), axis=0)
    unlabeled_data = np.concatenate((unlabeled_data, ood_data_used), axis=0)

    print(unlabeled_data.shape)
    unlabeled_data_save_path = os.path.join(working_path, 'data/rawdata-cleaned/unlabeled_content_raw_cleaned.npy')
    np.save(unlabeled_data_save_path, unlabeled_data)

    # construct the test data
    ind_test = data_dict['ind_test']
    ood_test = data_dict['ood_test']
    test_set = np.concatenate((ind_test,ood_test), axis=0)
    test_set_label = np.concatenate((np.ones(ind_test.shape[0]), np.zeros(ood_test.shape[0])),axis=0)

    print(test_set.shape, test_set_label.shape)
    testset_data_save_path = os.path.join(working_path, 'data/rawdata-cleaned/testset_content_raw_cleaned.npy')
    testset_label_save_path = os.path.join(working_path, 'data/finisheddata/testset_label.npy')
    np.save(testset_data_save_path, test_set)
    np.save(testset_label_save_path, test_set_label)



if __name__ == '__main__':
    label_transform()
    dataset_construct()