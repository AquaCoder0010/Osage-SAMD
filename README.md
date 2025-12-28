# Osage-SAMD (Signature Agnostic Malware Detection) through a Consistent Bidirectional GAN

## Project Overview
This project implements a **Consistent Bidirectional Generative Adversarial Network (CBiGAN)**, based on the research paper [**Enhanced Consistency Bidirectional GAN (CBiGAN)**](https://arxiv.org/abs/2506.07372), specifically optimized for unsupervised anomaly detection in malware images ($512 \times 512$). 

The model is trained exclusively on **benign (normal) data**. It learns the underlying distribution of "normalcy," allowing it to identify malware as deviations from this learned manifold based on reconstruction failure and feature discrepancies.


## 📊 Key Performance
Using the [DikeDataset](https://github.com/iosifache/DikeDataset) as a data source, malware binaries were converted to images through the [SABV](https://github.com/SABV-repo) package.

* **Peak AUC:** 0.82
* **Resolution:** $512 \times 512$ pixels
