ICDAR2015 :smile:
=====

- This repo is about to implement diffrent kinds of modern toolkits to detect and recognize texts in image. In the repo so far, I used DB and CRNN  (implemented in mmocr) and also, I tried a simple CRNN model written in Keras for recognizing text, which is much depended on [this](https://github.com/qjadud1994/CRNN-Keras) 

- Let's start with:

```
git clone https://github.com/manhph2211/ICDAR-2015.git 
cd ICDAR-2015
bash bash.sh
```

# Datasets

- This implement uses datasets from ICDAR-2015 competiton. You can download data from the [offical website](https://rrc.cvc.uab.es/?ch=4&com=downloads) (in the task 4.1 & 4.3). All zip files should be unzipped at `/data` 

# Text Localization

## Training (db-mmocr)

```
cd mmocr
python3 ./tools/train.py ../detector.py

```

# Text Recognition

## Training

- With crnn-mmocr:
```
cd mmocr
python3 ./tools/train.py ../recognizer.py

```

- With simple CRNN(Keras), just run `python3 train.py`
 






