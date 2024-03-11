# Introduction
This is the source code of TMM 2023 paper "Semi-Supervised Knowledge Distillation for Cross-Modal Hashing".

# Usage
For 64-bit hash codes:
## 1. Installation
- Python on v3.7
- CUDA on v11.4
- pip install -r requirements.txt

## 2. Train teacher model
(1) python SKDCH-it/xmedia/teacher/teacher_pretrain_i2t_64.py
(2) python SKDCH-it/xmedia/teacher/train_64_i2t.py

## 3. Generating pseudo label
(1) python SKDCH-it/xmedia/pseudo_label/updating_labels.py

## 4. Train student model
(1) python SKDCH-it/xmedia/student/teacher_pretrain_i2t_64.py
(2) python SKDCH-it/xmedia/student/train_64_i2t.py

# Citing
```
@inproceedings{su2023semi,
  title={Semi-supervised knowledge distillation for cross-modal hashing},
  author={Su, Mingyue and Gu, Guanghua and Ren, Xianlong and Fu, Hao and Zhao, Yao},
  journal={IEEE Transactions on Multimedia},
  volume={25},
  pages={662--675},
  year={2023},
  publisher={IEEE}
} 
```

# Contributing
Feel free to open an issue or contact us if you have any questions. ([guguanghua@ysu.edu.cn]())
