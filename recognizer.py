_base_ = []
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook')

    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# model
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)

model = dict(
    type='CRNNNet',
    preprocessor=None,
    backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)

train_cfg = None
test_cfg = None

# optimizer
optimizer = dict(type='Adadelta', lr=1.0)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[])
total_epochs = 5

# data
img_norm_cfg = dict(mean=[0.5], std=[0.5])

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=4,
        max_width=None,
        keep_aspect_ratio=True),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['filename', 'ori_shape', 'img_shape', 'valid_ratio']),
]

dataset_type = 'OCRDataset'

train_img_prefix = 'data/recognition'

train = dict(
    type=dataset_type,
    img_prefix=train_img_prefix+'/ch4_training_word_images_gt',
    ann_file= train_img_prefix + '/train_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

test_prefix = 'data/recognition'

test = dict(
    type=dataset_type,
    img_prefix=test_prefix+'/ch4_test_word_images_gt',
    ann_file=test_img_prefix + '/test_label.txt',
    loader=dict(
        type='HardDiskLoader',
        repeat=1,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    test_mode=True)


data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(type='ConcatDataset', datasets=[train]),
    val=dict(
        type='ConcatDataset',
        datasets=[test]),
    test=dict(
        type='ConcatDataset',
        datasets=[test]))

evaluation = dict(interval=1, metric='acc')

cudnn_benchmark = True