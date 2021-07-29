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

### Data
- [moable v1.21 (pytorch)](https://drive.google.com/drive/folders/1ZDerqTBeRvSWPshfODixjjvafpjjF9Mh?usp=sharing)


## Requirements

```bash
$ conda create -n MoAble python=3.6
$ conda activate MoAble
$ conda install numpy pandas requests scikit-learn
$ conda install -c rdkit rdkit
$ conda install -c conda-forge -c bioconda gseapy
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
Note that Pytorch has to be installed depending on the version of CUDA.

## Predict MoA

The source code is for predicting MoAs of novel compounds with MoAble.

```bash
$ python prediction.py 
```

GP embedding vectors, GP signature data, and pretrained model checkpoint are required to run the code. 

GP data can be downloaded from the Google Drive link in the resources section. 

Please contact Gwanghoon Jang (jghoon (at) korea.ac.kr) for downloading pretrained model checkpoint.

## License

This software is copyrighted by Data Mining and Information Systems Lab @ Korea University.

The source code and data can be used only for NON COMMERCIAL purposes.
