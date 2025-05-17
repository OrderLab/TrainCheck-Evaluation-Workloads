dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False
    )
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224),
    dict(type='Normalize',
         mean=[125.307, 122.961, 113.8575],
         std=[51.5865, 50.847, 51.255],
         to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='Normalize',
         mean=[125.307, 122.961, 113.8575],
         std=[51.5865, 50.847, 51.255],
         to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(type='Resize', size=224),
            dict(type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
            dict(type='Resize', size=224),
            dict(
                type='Normalize',
                mean=[125.307, 122.961, 113.8575],
                std=[51.5865, 50.847, 51.255],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=[
        dict(type='Resize', size=224),
        dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])
        ],
        test_mode=True
        )
)
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(
    type='ImageClassifier',
    backbone=dict(
    type='VGG',
    depth=19,
    num_classes=10,
    frozen_stages=1,
    out_indices=(-1, ),
    # init_cfg=dict(
    # type='Pretrained',
    # checkpoint=
    # 'checkpoints/vgg19_batch256_imagenet_20210208-e6920e4a.pth',
    # prefix='backbone'
    # )
    ),
    neck=None,
    head=dict(
    type='ClsHead',
    loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    topk=(1, 5))
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=20)
work_dir = './work_dirs/vgg19-finetune-cifar'
gpu_ids = range(0, 4)