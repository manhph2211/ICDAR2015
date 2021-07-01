
pip install mmcv-full==1.3.4

# install mmdetection
pip install mmdet==2.11.0

# install mmocr
git clone https://github.com/open-mmlab/mmocr.git
cd mmocr
pip install -r requirements.txt
pip install -v -e .  
mkdir checkpoints
cd ..

#-----------------------------------------------------------------------------

cd data/detection

## For images
mv ../ch4_training_images imgs/train
mv ../ch4_test_images imgs/test
## For annotations
mv ../ch4_training_localization_transcription_gt annotations/train
mv ../Challenge4_Test_Task1_GT annotations/test

wget https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json
wget https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json
wget -P ../../mmocr/checkpoints https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth

cd ../../

#-----------------------------------------------------------------------------


cd data/recognition

## For images
mv ../ch4_training_word_images_gt .
mv ../ch4_test_word_images_gt .


wget https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/train_label.txt
wget https://download.openmmlab.com/mmocr/data/mixture/icdar_2015/test_label.txt
wget -P ../../mmocr/checkpoints https://download.openmmlab.com/mmocr/textdet/dbnet/dbnet_r18_fpnc_sbn_1200e_icdar2015_20210329-ba3ab597.pth

cd ../../
