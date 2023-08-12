 # Data Augmentation for Few-Shot Named Entity Recognition Using ChatGPT with Transfer Learning(DACT)
 

### 1_aug_exam
<p align="center">
  <img src="./figures/1_aug_exam.jpg" width="400"/>
</p>
 

### 2_decode
<p align="center">
  <img src="./figures/3_decode.jpg" width="400"/>
</p>


### 2_model
<p align="center">
  <img src="./figures/2_model.jpg" />
</p>

## 1. Environments

```
- python (3.8.12)
- cuda (11.4)
```

## 2. Dependencies

```
- numpy (1.21.4)
- torch (1.10.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
```

## 3. Dataset

We provide some datasets processed in our code.

## 4. Preparation

- Augment using ChatGPT based on Few-shot original dataset (Already given in the code)
- Transfer weights from pre-trained relevant dataset.
- When training on augmented Few-shot original dataset, transfer weights from pre-trained relevant data.

## 5. Training

```bash
- Training Transfer Weights
>> python main-All_BC5CDR-disease.py
>> python main-All_NCBI.py
>> python main-All_BioNLP11EPI-IOBES.py
>> python main-All_BioNLP13GE-IOBES.py

- Copying Transfer Weights for Different Datasets. For all datasets: 5-1,5-2,5-3,5-4,5-5,20-1,20-2,20-3,20-4,20-5,50-1,50-2,50-3,50-4,50-5, follow the same training method as 20-1:
- "train_name" and "dev_name" represent the names of the datasets used for training and development, respectively.
- Taking the training of the BC5CDR-disease dataset as an example, where "All_NCBI.pt" represents the transfer weights saved during training using the command 'python main-All_NCBI.py'.

-Train the BC5CDR-disease dataset.
>> cp All_NCBI.pt NCBI-20-1.pt
>> python main.py --config ./config/All_BC5CDR-disease-20-1.json --train_name train_best_clear --dev_name dev_best_clear

-Train the NCBI dataset.
>> cp All_BC5CDR-disease.pt BC5-20-1.pt
>> python main.py --config ./config/All_BC5CDR-disease-20-1.json --train_name train_best_clear --dev_name dev_best_clear

-Train the BioNLP11EPI dataset.
>> cp All_BioNLP13GE-IOBES.pt Bio13-20-1.pt
>> python main.py --config ./config/All_BC5CDR-disease-20-1.json --train_name train_best_clear --dev_name dev_best_clear

-Train the BioNLP13GE dataset.
>> cp All_BioNLP11EPI-IOBES.pt Bio11-20-1.pt
>> python main.py --config ./config/All_BC5CDR-disease-20-1.json --train_name train_best_clear --dev_name dev_best_clear

If you want to train datasets other than 20-1 shot, you just need to replace "20-1" in the above statement with any of the following: 5-1, 5-2, 5-3, 5-4, 5-5, 20-1, 20-2, 20-3, 20-4, 20-5, 50-1, 50-2, 50-3, 50-4, 50-5.
```




