# adversarial-ood-detection
The codes and appendices of paper 'Practical and Efficient Out-of-Domain Detection with Adversarial Learning', ACM SAC' 22. 
This demo code used [CLINC](https://github.com/clinc/oos-eval) dataset as the example.


## Running method:
You can run the program by simply executing
```
bash run.sh
```
The final available OOD classifier, along with its required tokenizer, embedding matrix and setting-file, will be stored in the `final-ood-detector_model_DATE` folder.

You may also run the program by
```
python main.py --clean_tempfiles='true'
```
If you want to keep the preprocessed data and the trained intermediate networks, you can change the option `clean_tempfiles` to `false` (the default is `true`). 

### Notes:
1. We recommend to run the experiment in an anaconda environment, for it pre-installed many common libraries. You may need to install the other necessary libraries before the experiment by running `$ pip install -r requirements.txt` .
2. Because this study used the [fasttext](https://fasttext.cc/docs/en/crawl-vectors.html) word embedding, it may take a relatively long time to download the embedding file when you run the code for the first time.
3. The log file recording the full training procedure is saved in `log_file.txt`.

## Reference
If you find our study useful, please consider citing us:
```
@inproceedings{wang2022practical,
  title={Practical and efficient out-of-domain detection with adversarial learning},
  author={Wang, Bo and Mine, Tsunenori},
  booktitle={Proceedings of the 37th ACM/SIGAPP Symposium on Applied Computing},
  pages={853--862},
  year={2022}
}
```


