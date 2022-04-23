#!/usr/bin/env python
# -*- coding:utf-8 -*-

from adv_ood_detect_framework import pre_a_json_to_npy, pre_b_one_text_clean, pre_b_two_label_trans_dataset_cons, pre_c_nltk_cut_pad, pre_d_tokenizer_embd_matrix
from adv_ood_detect_framework import a_ind_classifier_train, b_one_autoencoder, b_two_latent_code_encode, c_one_gan_ood_train, c_two_latent_code_decode, d_ood_detector_train, e_test_result_check

import argparse
import os, sys, shutil
import json
from datetime import datetime


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--clean_tempfiles', type=str, default = 'true')
args = parser.parse_args()



now = datetime.now()
date_str = now.strftime("%Y_%m_%d_%H_%M")
print(date_str, 'Start training OOD detector models.')
print('\n')

# Logger records the running logs
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()  

    def flush(self):
        self.log.flush()

sys.stdout = Logger("./log_file.txt")


print('#########################################')
print('# Writing basic info to exp_settings.json #')
print('#########################################')


settings_json = {"model foldername": date_str}
with open('exp_settings.json', 'w') as f:
    json.dump(settings_json, f)
print('Complete!')
print('\n')





print('#########################################')
print('#       Making temp folders...          #')
print('#########################################')
print('\n')
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  
        os.makedirs(path)  
        print("-- temp folder ", path, " created --")
    else:
        print("---  the folder already exists!  ---")


def rm_dir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        print("---  folder does not exist  ---")
    else:
        shutil.rmtree(path)
        print("-- temp folder ", path, " removed --")



working_path = os.path.abspath('.') 
mkdir(os.path.join(working_path, 'data'))
mkdir(os.path.join(working_path, 'data/rawdata'))
mkdir(os.path.join(working_path, 'data/rawdata-cleaned'))
mkdir(os.path.join(working_path, 'data/finisheddata'))
mkdir(os.path.join(working_path, 'data/latentcodedata'))
mkdir(os.path.join(working_path, 'data/generateddata'))
mkdir(os.path.join(working_path, 'otherdatafiles'))
mkdir(os.path.join(working_path, 'trainedmodels'))
mkdir(os.path.join(working_path, 'final-ood-detector'))
print('Complete!')
print('\n')


print('#########################################')
print('#         The training starts!          #')
print('#########################################')
print('\n')

print('#########################################')
print('#     Preprocessing the texts...        #')
print('#########################################')
pre_a_json_to_npy.read_json_to_npy()
pre_b_one_text_clean.text_clean()
pre_b_two_label_trans_dataset_cons.label_transform()
pre_b_two_label_trans_dataset_cons.dataset_construct()
pre_c_nltk_cut_pad.truncate_decide_cut()
pre_d_tokenizer_embd_matrix.tokenizer_matrix_build()
print('Complete!')
print('\n')

print('#########################################')
print('#         Training IND classifer        #')
print('#########################################')
a_ind_classifier_train.train_ind_classifier()
print('Complete!')
print('\n')

print('#########################################')
print('#          Training Autoencoder         #')
print('#########################################')
b_one_autoencoder.train_autoencoder()
print('# Encoding data to vector with encoder  #')
b_two_latent_code_encode.encode_data_to_vector()
print('Complete!')
print('\n')

print('#########################################')
print('#          Training GAN network         #')
print('#########################################')
c_one_gan_ood_train.gan_training()
print('#      Decoding generated vectors...    #')
c_two_latent_code_decode.decode_generate_code()
print('Complete!')
print('\n')

print('#########################################')
print('#          Training OOD detector       #')
print('#########################################')
d_ood_detector_train.train_ood_detector()
print('Complete!')
print('\n')


print('#########################################')
print('#      Preforming final tests...        #')
print('#########################################')
e_test_result_check.check_result_on_test()
print('Complete!')
print('\n')

if args.clean_tempfiles == 'true':
    rm_dir(os.path.join(working_path, 'data'))
    rm_dir(os.path.join(working_path, 'otherdatafiles'))
    rm_dir(os.path.join(working_path, 'trainedmodels'))

shutil.copy('exp_settings.json', 'final-ood-detector/')
os.rename(os.path.join(working_path, 'final-ood-detector'), os.path.join(working_path, ('final-ood-detector_model_' + date_str)))
print('Training complete, the OOD detector, as well as tokenizer, embedding matrix and settings are saved in folder: ')
print(os.path.join(working_path, ('final-ood-detector_model_' + date_str)))







