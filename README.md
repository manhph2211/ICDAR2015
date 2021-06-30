ICDAR2015 :smile:
=====

- This repo is about to implement diffrent kinds of modern toolkits to detect and regconize texts in image.

- Let's start with:

```
git clone https://github.com/manhph2211/ICDAR-2015.git 
cd ICDAR-2015
```

# Datasets

- This implement uses datasets from ICDAR-2015 competiton. You can download data from the [offical website](https://rrc.cvc.uab.es/?ch=4&com=downloads) (in the task 4.1 & 4.3). All zip files should be unzipped at `/data` 

# Text Localization


## Training

```
bash bash.sh
cd mmocr
python3 ./tools/train.py ../trainer.py

```

# Text Regconition

## Datasets

- This implement uses datasets from ICDAR-2015 competiton. You can download data from the [offical website](https://rrc.cvc.uab.es/?ch=4&com=downloads) (in the task 4.1). All zip files should be unzipped at `/ICDAR-2015` 

## Training

```
bash bash.sh
cd mmocr
python3 ./tools/train.py ../trainer.py

```






