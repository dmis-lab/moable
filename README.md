# MoAble

A Pytorch Implementation of paper

Predicting mechanism of action of novelcompounds using compound structure andtranscriptomic signature co-embedding

Jang and Park et al., 2021

## Abstract

Identifying mechanism of actions (MoA) of novel compounds is crucial in drug discovery. Careful understanding of MoA can avoid potential side effects of drug candidates. Efforts have been madeto identify MoA using the transcriptomic signatures induced by compounds. However, those approachesfail to reveal MoAs in the absence of actual compound signatures.

We present MoAble, which predicts MoAs without requiring compound signatures. We train adeep learning-based co-embedding model to map compound signatures and compound structure intothe same embedding space. The model generates low-dimensional compound signature representationfrom the compound structure. To predict MoAs, pathway enrichment analysis is performed based on theconnectivity between embedding vectors of compounds and those of genetic perturbation. Results showthat MoAble is comparable to the methods that use actual compound signatures. We demonstrate thatMoAble can be used to reveal MoAs of novel compounds without measuring compound signatures withthe same prediction accuracy as measuring it.

## Overview of MoAble

![overview](https://user-images.githubusercontent.com/56992294/106699777-dbf52a80-6626-11eb-824a-cf41530380d5.png)


## Resources

### Pretrained model & Data
- [moable v1.1 (pytorch)](https://drive.google.com/drive/folders/1ZDerqTBeRvSWPshfODixjjvafpjjF9Mh?usp=sharing)
