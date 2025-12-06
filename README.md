## [An unsupervised long-tailed fine-grained few-shot fault diagnosis method based on feature space adjustment strategy](https://doi.org/10.1016/j.ress.2025.111998)
In the operation of mechanical equipment, fault data often display an implicit long-tailed distribution, making manual labeling labor-intensive. Moreover, traditional fault diagnosis methods struggle to identify fault causes from a fine-grained perspective without labels. Based on those, an unsupervised long-tailed fine-grained few-shot fault diagnosis method, USL-F4SA, is proposed, which dynamically enhances batches through a queue mechanism and applies indicators and auxiliary sets to construct a class balance mechanism in the feature space. Initially, K-means clustering identifies cluster centers of the original training set, the sparsity of clustering features is used to detect tail classes. Samples with high cosine similarity to the original training set are selected from an auxiliary dataset to balance the distribution of tail classes. Then, a dynamic queue is introduced to find the nearest neighbor for each sample, which is added as a positive sample to the batch. The optimal transport algorithm samples the queue, ensuring optimal assignment between current batch embeddings and the queue, allowing dynamic adaptation to new data. Finally, an indicator assigns indexes to all samples, ensuring correct sample pairing and distinguishing samples from different distributions. Validated on public datasets, USL-F4SA outperforms state-of-the-art methods in predicting unseen fine-grained faults. 

# If it is helpful for your research, please kindly cite this work:
﻿
```html

@article{bai2023effectiveness,
  title={On the effectiveness of out-of-distribution data in self-supervised long-tail learning},
  author={Bai, Jianhong and Liu, Zuozhu and Wang, Hualiang and Hao, Jin and Feng, Yang and Chu, Huanpeng and Hu, Haoji},
  journal={arXiv preprint arXiv:2306.04934},
  year={2023}
}

@article{ZHAO2026111998,
title = {An unsupervised long-tailed fine-grained few-shot fault diagnosis method based on feature space adjustment strategy},
journal = {Reliability Engineering & System Safety},
volume = {268},
pages = {111998},
year = {2026},
issn = {0951-8320},
doi = {https://doi.org/10.1016/j.ress.2025.111998},
url = {https://www.sciencedirect.com/science/article/pii/S0951832025011974},
author = {Zhiqian Zhao and Yinghou Jiao and Yeyin Xu and Xiang Zhang and Runchao Zhao and Zhaobo Chen}
}
