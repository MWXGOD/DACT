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

- 根据小样本原始数据，使用ChatGPT进行增强（code中已经给出）
- 预训练相关数据集的迁移权重
- 在训练增强后的小样本数据时，迁移已经训练好的相关数据权重

## 5. Training

```bash
- 训练迁移权重
>> python main.py --config ./config/example.json
```
## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 7. Citation

If you use this work or code, please kindly cite this paper:

```
@inproceedings{li2022unified,
  title={Unified named entity recognition as word-word relation classification},
  author={Li, Jingye and Fei, Hao and Liu, Jiang and Wu, Shengqiong and Zhang, Meishan and Teng, Chong and Ji, Donghong and Li, Fei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={10},
  pages={10965--10973},
  year={2022}
}
```



