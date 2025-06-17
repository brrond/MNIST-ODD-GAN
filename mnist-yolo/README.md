# MNIST Object Detection with YOLO

For the purpose of verifying generated dataset some base models for object detection must be trained first.

train yolo model:
```
yolo train model=yolo11s.pt data=mnist.yaml epochs=1000 patience=100 batch=16 imgsz=320 save=True optimizer=adam seed=42 deterministic=True cos_lr=True amp=False lr0=0.005 lrf=0.01 momentum=0.937 warmup_epochs=5 dropout=0.3 plots=True
```

Output:
```
Ultralytics 8.3.155  Python-3.9.21 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce GTX 1660 Ti, 6144MiB)
engine\trainer: agnostic_nms=False, amp=False, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=True, cutmix=0.0, data=mnist.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.3, dynamic=False, embed=None, epochs=1000, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=320, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.005, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11s.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train6, nbs=64, nms=False, opset=None, optimize=False, optimizer=adam, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=runs\detect\train6, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=42, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=5, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=10

                   from  n    params  module                                       arguments
  0                  -1  1       928  ultralytics.nn.modules.conv.Conv             [3, 32, 3, 2]
  1                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  2                  -1  1     26080  ultralytics.nn.modules.block.C3k2            [64, 128, 1, False, 0.25]
  3                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
  4                  -1  1    103360  ultralytics.nn.modules.block.C3k2            [128, 256, 1, False, 0.25]
  5                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
  6                  -1  1    346112  ultralytics.nn.modules.block.C3k2            [256, 256, 1, True]
  7                  -1  1   1180672  ultralytics.nn.modules.conv.Conv             [256, 512, 3, 2]
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 13                  -1  1    443776  ultralytics.nn.modules.block.C3k2            [768, 256, 1, False]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 16                  -1  1    127680  ultralytics.nn.modules.block.C3k2            [512, 128, 1, False]
 17                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 19                  -1  1    345472  ultralytics.nn.modules.block.C3k2            [384, 256, 1, False]
 20                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 22                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]
 23        [16, 19, 22]  1    823278  ultralytics.nn.modules.head.Detect           [10, [128, 256, 512]]
YOLO11s summary: 181 layers, 9,431,662 parameters, 9,431,646 gradients, 21.6 GFLOPs

Transferred 493/499 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
train: Fast image access  (ping: 0.00.0 ms, read: 43.123.4 MB/s, size: 5.3 KB)
train: Scanning D:\repos\MNIST-ODD-GAN\MNIST-ObjectDetection\data\mnist_detection\train\labels.cache... 100 images, 0 backgrounds, 0 corrupt: 100%|██████████| 100/100 [00:00<?, ?it/s]
val: Fast image access  (ping: 0.10.0 ms, read: 27.815.9 MB/s, size: 4.8 KB)
val: Scanning D:\repos\MNIST-ODD-GAN\MNIST-ObjectDetection\data\mnist_detection\validation\labels.cache... 100 images, 0 backgrounds, 0 corrupt: 100%|██████████| 100/100 [00:00<?, ?it/s]
Plotting labels to runs\detect\train6\labels.jpg...
optimizer: Adam(lr=0.005, momentum=0.937) with parameter groups 81 weight(decay=0.0), 88 weight(decay=0.0005), 87 bias(decay=0.0)
Image sizes 320 train, 320 val
Using 8 dataloader workers
Logging results to runs\detect\train6
Starting training for 1000 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     1/1000      2.23G      1.197      3.457      1.029          7        320: 100%|██████████| 7/7 [00:05<00:00,  1.19it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:02<00:00,  1.98it/s]
                   all        100        488      0.145      0.449      0.153        0.1

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     2/1000      2.63G     0.9466      2.364     0.9414         19        320: 100%|██████████| 7/7 [00:01<00:00,  4.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  2.66it/s]
                   all        100        488      0.142      0.636      0.182      0.126

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     3/1000      2.69G     0.9949        2.1     0.9885         23        320: 100%|██████████| 7/7 [00:01<00:00,  4.90it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  2.86it/s]
                   all        100        488      0.288       0.56      0.259      0.175

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     4/1000      2.69G     0.9222       1.83     0.9578         28        320: 100%|██████████| 7/7 [00:01<00:00,  4.79it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.12it/s]
                   all        100        488       0.19      0.618      0.277      0.182

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     5/1000      2.69G     0.9945      1.791      1.021          7        320: 100%|██████████| 7/7 [00:01<00:00,  5.07it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.52it/s]
                   all        100        488     0.0618      0.153     0.0643     0.0343

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     6/1000      2.71G      1.036       1.59      1.041         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  2.97it/s]
                   all        100        488   3.57e-05    0.00678   1.85e-05   5.75e-06

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     7/1000      2.77G     0.9853      1.495      1.017         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  2.90it/s]
                   all        100        488          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     8/1000      2.81G      1.017      1.412      1.017         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.77it/s]
                   all        100        488          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     9/1000      2.85G     0.9741      1.364      1.017         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.86it/s]
                   all        100        488          0          0          0          0

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    10/1000      2.87G      1.028      1.398      1.004         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.65it/s]
                   all        100        488      0.168      0.116     0.0661     0.0295

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    11/1000      2.87G      1.069      1.423      1.061         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.94it/s]
                   all        100        488      0.358      0.278      0.248       0.15

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    12/1000      2.87G      1.051      1.336      1.046         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.79it/s]
                   all        100        488     0.0381     0.0673      0.032    0.00985

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    13/1000      2.87G      1.013      1.279      1.028         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.69it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.44it/s]
                   all        100        488      0.153     0.0707     0.0293     0.0102

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    14/1000      2.87G     0.9915      1.288      1.014         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.523     0.0252    0.00736    0.00159

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    15/1000      2.87G      1.013      1.283       1.02         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.24it/s]
                   all        100        488      0.133      0.114     0.0187     0.0061

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    16/1000      2.87G      1.038      1.264      1.042         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.05it/s]
                   all        100        488      0.545      0.258      0.287       0.17

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    17/1000      2.89G     0.9746      1.267     0.9951         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.10it/s]
                   all        100        488      0.603      0.342      0.385      0.257

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    18/1000      2.93G     0.9497      1.245      1.011         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.13it/s]
                   all        100        488      0.505      0.432      0.475      0.306

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    19/1000      2.93G     0.9671      1.211      1.009         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.22it/s]
                   all        100        488      0.378      0.363      0.304      0.207

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    20/1000      2.93G     0.9053      1.118      0.964         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.434      0.315      0.226      0.141

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    21/1000      2.93G     0.9227      1.196     0.9771         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.23it/s]
                   all        100        488      0.474      0.334      0.315      0.207

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    22/1000      2.93G     0.9047      1.054     0.9631         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.38it/s]
                   all        100        488      0.523      0.592      0.583      0.394

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    23/1000      2.93G      0.898      1.115     0.9831         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488        0.4      0.583       0.44      0.302

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    24/1000      2.93G     0.9247      1.168     0.9822         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.602      0.677        0.7      0.495

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    25/1000      2.93G      0.911      1.014     0.9379         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.38it/s]
                   all        100        488      0.588      0.659      0.666      0.493

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    26/1000      2.93G     0.9332      1.021     0.9695         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.491      0.601      0.545      0.396

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    27/1000      2.93G     0.8683      1.035     0.9692         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.732      0.649      0.745      0.554

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    28/1000      2.93G     0.8655     0.9696     0.9784         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.09it/s]
                   all        100        488      0.668      0.633      0.715      0.517

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    29/1000      2.93G     0.8791     0.9454     0.9568         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.38it/s]
                   all        100        488      0.689      0.658      0.731      0.527

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    30/1000      2.93G     0.9369     0.9866     0.9887         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488       0.76      0.493      0.651      0.497

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    31/1000      2.93G     0.8151     0.9376     0.9374         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.04it/s]
                   all        100        488      0.685      0.681      0.701      0.519

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    32/1000      2.93G     0.8317      0.909     0.9491         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]
                   all        100        488      0.704      0.726      0.795      0.596

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    33/1000      2.93G     0.8315     0.8853     0.9382         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.764      0.702      0.793      0.593

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    34/1000      2.93G     0.7897     0.8311     0.9279         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.16it/s]
                   all        100        488      0.788      0.677      0.803      0.617

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    35/1000      2.93G     0.7867     0.9065     0.9221         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.788      0.698      0.809      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    36/1000      2.93G      0.788     0.8492     0.9346         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.733      0.772       0.82      0.635

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    37/1000      2.93G     0.7934     0.8616     0.9383         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488      0.782       0.71      0.799       0.62

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    38/1000      2.93G      0.764     0.8524     0.9455         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.749      0.705      0.818      0.641

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    39/1000      2.93G     0.7473     0.8571     0.9503         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.762      0.651      0.787      0.618

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    40/1000      2.93G     0.7522     0.8003     0.9278         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488       0.76      0.747      0.816      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    41/1000      2.93G     0.7791     0.8436     0.9308         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.741      0.716      0.805      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    42/1000      2.93G     0.8064     0.8164     0.9238         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]
                   all        100        488      0.799      0.681      0.786      0.615

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    43/1000      2.93G     0.8372     0.8967     0.9457         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.20it/s]
                   all        100        488      0.697      0.655      0.741      0.546

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    44/1000      2.93G     0.7968     0.8162     0.9201         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.763      0.621      0.737      0.582

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    45/1000      2.93G     0.7205     0.8068     0.8855         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.21it/s]
                   all        100        488      0.742      0.718      0.794      0.627

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    46/1000      2.93G     0.7669     0.8138     0.9328         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.775      0.745      0.819      0.654

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    47/1000      2.93G     0.8271     0.7892      0.946         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488      0.754      0.719       0.81      0.648

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    48/1000      2.93G     0.8046     0.8506     0.9329         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.706      0.714      0.778       0.61

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    49/1000      2.93G     0.7725     0.7981     0.9223         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.824      0.741      0.826      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    50/1000      2.93G     0.7786     0.8104     0.9557         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.837      0.682      0.844      0.657

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    51/1000      2.93G     0.7086     0.7724     0.9255         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.31it/s]
                   all        100        488      0.729       0.73      0.824      0.665

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    52/1000      2.93G     0.7433     0.7509     0.9719         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488       0.77      0.719      0.804      0.659

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    53/1000      2.93G     0.7055     0.7428     0.9106         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.804       0.75      0.813      0.659

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    54/1000      2.93G     0.7402     0.7379     0.9147         42        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.751       0.74      0.794       0.64

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    55/1000      2.93G     0.7284      0.726     0.9239         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488      0.761      0.777      0.829       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    56/1000      2.93G     0.6861     0.7121     0.9138         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.769      0.786      0.855      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    57/1000      2.93G     0.6853     0.6977     0.9163         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.851      0.751      0.853      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    58/1000      2.93G     0.6974     0.7053      0.917         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.844       0.75       0.86      0.714

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    59/1000      2.93G     0.6972     0.7302     0.9003         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.864      0.716      0.865      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    60/1000      2.93G     0.7258     0.7628       0.93         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.809      0.738      0.837      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    61/1000      2.93G     0.6569     0.7076     0.8996         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.765       0.72      0.817       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    62/1000      2.93G     0.6903     0.7472     0.9082         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.801      0.703      0.822      0.666

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    63/1000      2.93G     0.6934     0.6909     0.9224         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488      0.824      0.732      0.843      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    64/1000      2.93G     0.7038     0.6678      0.886         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.804      0.746      0.845      0.666

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    65/1000      2.93G     0.7016     0.6727     0.9075         51        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.822      0.768      0.864        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    66/1000      2.93G     0.6978     0.6521      0.891         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.844      0.787       0.87      0.693

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    67/1000      2.93G     0.7367     0.7145     0.9217         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.843      0.753      0.864      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    68/1000      2.93G     0.6939        0.7     0.9282         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.885      0.745      0.869      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    69/1000      2.93G     0.6904     0.6721     0.8859         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.842      0.801      0.876      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    70/1000      2.93G     0.6875     0.6969     0.9171         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.795      0.779      0.865      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    71/1000      2.93G     0.6675     0.6368     0.8923         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.53it/s]
                   all        100        488      0.789      0.791      0.874      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    72/1000      2.93G     0.6605     0.7025     0.9047         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.806      0.818      0.869      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    73/1000      2.93G     0.6237     0.6417     0.8899         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.782      0.836      0.872       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    74/1000      2.93G     0.6657     0.6411     0.8927         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.17it/s]
                   all        100        488      0.766      0.734       0.82      0.663

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    75/1000      2.93G     0.6668      0.697     0.8818         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.88it/s]
                   all        100        488      0.775      0.708      0.817      0.654

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    76/1000      2.93G     0.6759     0.6434     0.8876         35        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.83it/s]
                   all        100        488      0.777      0.725      0.829      0.673

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    77/1000      2.93G     0.6917     0.6954     0.9044         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.86it/s]
                   all        100        488      0.789      0.745      0.847      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    78/1000      2.93G     0.6597     0.6661     0.9093         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.16it/s]
                   all        100        488      0.807      0.817      0.877      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    79/1000      2.93G     0.7017     0.6917     0.9095         46        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.53it/s]
                   all        100        488      0.816      0.807      0.874      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    80/1000      2.93G     0.6966      0.686     0.9091         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.817      0.788       0.88      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    81/1000      2.93G     0.7166     0.7312     0.9015         10        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.881       0.72       0.86      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    82/1000      2.93G     0.6838     0.6623     0.8981         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.892      0.744      0.873      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    83/1000      2.93G     0.6534     0.6386     0.8696         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488       0.86      0.729      0.853      0.706

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    84/1000      2.93G     0.6322     0.6678     0.8905         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.31it/s]
                   all        100        488      0.807      0.792      0.869      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    85/1000      2.93G     0.6464      0.645     0.9055         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.809      0.814      0.881      0.728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    86/1000      2.93G     0.6567     0.6246      0.885         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.847      0.773      0.874      0.717

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    87/1000      2.93G     0.6462     0.6436     0.9004         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.21it/s]
                   all        100        488      0.871      0.757      0.889      0.746

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    88/1000      2.93G     0.6183     0.5901     0.8992         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.854      0.765      0.882      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    89/1000      2.93G     0.6645     0.6561     0.9134         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.773      0.811      0.872      0.713

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    90/1000      2.93G     0.6075      0.588      0.883         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488       0.86      0.767      0.864      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    91/1000      2.93G     0.6233     0.5745     0.8796         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.818      0.749       0.84       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    92/1000      2.93G     0.6429     0.6284     0.9024         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]
                   all        100        488      0.926      0.738       0.88      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    93/1000      2.93G      0.638     0.5982     0.8943         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.853      0.784      0.878      0.733

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    94/1000      2.93G      0.624     0.6164     0.8781         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.862      0.758      0.862       0.71

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    95/1000      2.93G     0.6573     0.6158     0.8925         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.849      0.773      0.868      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    96/1000      2.93G     0.6225     0.6746     0.9255         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.836      0.778       0.86      0.721

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    97/1000      2.93G     0.6319     0.6359     0.8951         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.21it/s]
                   all        100        488      0.859      0.766      0.866      0.716

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    98/1000      2.93G     0.6345     0.5547     0.8932         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.87it/s]
                   all        100        488      0.874       0.79      0.875      0.729

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    99/1000      2.93G     0.6244     0.6127     0.9053         37        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.861      0.789       0.88       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   100/1000      2.93G     0.6011     0.5877     0.8888         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.871      0.773      0.876       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   101/1000      2.93G     0.6178      0.551     0.8586         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.853       0.78      0.877       0.73

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   102/1000      2.93G     0.6422     0.6355     0.8888         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.829        0.8      0.863       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   103/1000      2.93G     0.6437     0.5968     0.9041         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.798      0.795      0.849      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   104/1000      2.93G     0.6199     0.5628     0.8806         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.822      0.764      0.841      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   105/1000      2.93G     0.5837     0.5628      0.896         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.838      0.777       0.86      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   106/1000      2.93G     0.6432     0.5869     0.8947         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.08it/s]
                   all        100        488      0.856      0.779       0.87      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   107/1000      2.93G     0.6055     0.5502     0.8837         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.903      0.787      0.894      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   108/1000      2.93G     0.6245     0.5798     0.8763         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.98it/s]
                   all        100        488      0.877      0.793      0.894      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   109/1000      2.93G     0.6078      0.555     0.8715         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.854      0.802      0.888       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   110/1000      2.93G     0.6261     0.5689     0.8927         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.845      0.796      0.871      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   111/1000      2.93G     0.6139     0.5769     0.8764         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.861      0.767      0.861      0.712

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   112/1000      2.93G     0.5973     0.5931     0.8648         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.26it/s]
                   all        100        488      0.865      0.768      0.875      0.736

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   113/1000      2.93G     0.6003     0.5903     0.8898         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.835        0.8      0.872      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   114/1000      2.93G     0.5582     0.5375     0.8709         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.865       0.77      0.882      0.746

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   115/1000      2.93G     0.5776     0.5233     0.8639         38        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.874      0.778      0.884      0.749

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   116/1000      2.93G     0.5839     0.5312     0.8549         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.839      0.792      0.885      0.739

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   117/1000      2.93G     0.6255     0.6009     0.8957         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.843      0.815      0.892      0.745

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   118/1000      2.93G     0.6333     0.5826     0.8795         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488      0.871      0.811      0.892      0.764

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   119/1000      2.93G     0.5616     0.5464     0.8791         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.862      0.792      0.881      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   120/1000      2.93G     0.5978     0.5501     0.8748         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.813       0.81      0.874      0.727

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   121/1000      2.93G     0.6318     0.5904     0.8966         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.839      0.792      0.885      0.743

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   122/1000      2.93G     0.6052     0.5564     0.8666         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.54it/s]
                   all        100        488      0.821      0.765      0.861      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   123/1000      2.93G     0.6276     0.5783     0.9055         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.796      0.756      0.847      0.709

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   124/1000      2.93G     0.6081     0.5798      0.888         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.53it/s]
                   all        100        488      0.791      0.764      0.826      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   125/1000      2.93G     0.6246      0.613     0.8758         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.843      0.789      0.851      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   126/1000      2.93G     0.5887     0.5278     0.8592         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488       0.82      0.819      0.875      0.718

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   127/1000      2.93G     0.5987      0.533     0.8758         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.861      0.814      0.882      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   128/1000      2.93G     0.5953     0.5474     0.8974         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.832      0.819      0.873      0.726

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   129/1000      2.93G     0.6268     0.5599     0.8901         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.24it/s]
                   all        100        488      0.836      0.785      0.853      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   130/1000      2.93G     0.6071     0.5431      0.868         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.23it/s]
                   all        100        488      0.836      0.729      0.847      0.689

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   131/1000      2.93G     0.5806     0.5698     0.8894         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.20it/s]
                   all        100        488      0.798      0.745      0.838      0.675

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   132/1000      2.93G     0.5941      0.564     0.8877         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488      0.723      0.691       0.77      0.582

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   133/1000      2.93G     0.5865     0.5199     0.8859         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.695      0.736      0.782      0.606

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   134/1000      2.93G     0.5646     0.5339     0.8709         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.774      0.763      0.843      0.703

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   135/1000      2.93G      0.557      0.513     0.8752         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.806      0.805      0.872      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   136/1000      2.93G     0.5501     0.5011     0.8535         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.807      0.807      0.863      0.715

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   137/1000      2.93G     0.5595     0.5181     0.8625         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.23it/s]
                   all        100        488      0.863      0.812      0.882      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   138/1000      2.93G     0.5975     0.5446     0.8674         35        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.834      0.821      0.884      0.757

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   139/1000      2.93G     0.5836     0.5327     0.8593         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.58it/s]
                   all        100        488      0.891      0.779      0.891      0.757

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   140/1000      2.93G      0.554     0.5013     0.8582         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488       0.86      0.814      0.896      0.768

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   141/1000      2.93G     0.5606     0.5283     0.8786         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.845      0.822      0.896       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   142/1000      2.93G     0.5643     0.4965     0.8652         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.821      0.851      0.888      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   143/1000      2.93G     0.5673      0.527     0.8694         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488       0.84      0.807      0.883      0.767

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   144/1000      2.93G     0.5734     0.5547     0.8752         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488       0.86      0.775      0.881      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   145/1000      2.93G     0.5762     0.5666     0.8727         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.26it/s]
                   all        100        488      0.872      0.776      0.884      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   146/1000      2.93G     0.5653     0.5381     0.8647         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.858      0.757      0.873      0.743

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   147/1000      2.93G     0.5595     0.5339     0.8774         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.879      0.785      0.885      0.764

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   148/1000      2.93G     0.5765     0.5738     0.8822         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.59it/s]
                   all        100        488      0.873      0.787      0.891       0.75

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   149/1000      2.93G     0.5849     0.5496     0.8656         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.865      0.821      0.902       0.76

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   150/1000      2.93G     0.5775      0.573     0.8809         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.846      0.824      0.899      0.766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   151/1000      2.93G     0.5596     0.5216     0.8644         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.871      0.792      0.887       0.76

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   152/1000      2.93G     0.5639     0.5351      0.858         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.825      0.826      0.888      0.748

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   153/1000      2.93G     0.5852     0.5476     0.8741         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.861      0.817      0.889      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   154/1000      2.93G     0.6252     0.5923      0.891         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.21it/s]
                   all        100        488      0.849      0.815      0.881       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   155/1000      2.93G     0.5728     0.5234     0.8589         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.859      0.812      0.885      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   156/1000      2.93G     0.5975     0.5336     0.8638         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.22it/s]
                   all        100        488      0.856       0.79       0.88      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   157/1000      2.93G     0.5562     0.5148     0.8609         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.853      0.806      0.886      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   158/1000      2.93G     0.5537     0.5089     0.8547         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.863      0.798       0.89      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   159/1000      2.93G     0.5619     0.5069     0.8878         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.26it/s]
                   all        100        488       0.84      0.834        0.9      0.761

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   160/1000      2.93G      0.566     0.5161     0.8426         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.773      0.818      0.883      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   161/1000      2.93G     0.6215     0.5747     0.9107         10        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.807       0.83      0.893       0.75

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   162/1000      2.93G     0.5773     0.4909      0.863         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.884      0.808      0.899      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   163/1000      2.93G     0.5649     0.5005     0.8629         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.864      0.825      0.888      0.748

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   164/1000      2.93G     0.5544     0.4936     0.8574         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.848      0.829      0.894      0.749

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   165/1000      2.93G     0.5797     0.4945     0.8526         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488       0.85      0.829      0.901      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   166/1000      2.93G     0.5726      0.506     0.8731         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.868      0.832      0.911       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   167/1000      2.93G     0.5903      0.495     0.8746         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.55it/s]
                   all        100        488      0.881      0.801      0.904      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   168/1000      2.93G     0.5272      0.481     0.8588         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.843      0.826      0.893      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   169/1000      2.93G     0.5876     0.5153     0.8664         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488       0.87      0.816      0.896      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   170/1000      2.93G     0.5623     0.5176     0.8613         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.857      0.821      0.901      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   171/1000      2.93G       0.57     0.5043     0.8672         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.878        0.8      0.909       0.77

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   172/1000      2.93G     0.5869     0.5281     0.8827         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.882       0.79      0.893      0.746

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   173/1000      2.93G     0.5519     0.4854     0.8613         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.887      0.802      0.893      0.743

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   174/1000      2.93G     0.5434     0.4925     0.8587         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.877      0.842      0.904      0.765

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   175/1000      2.93G     0.5401     0.5051     0.8556         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.26it/s]
                   all        100        488      0.887      0.827      0.911      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   176/1000      2.93G     0.5417      0.477     0.8749         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.875      0.836      0.911      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   177/1000      2.93G     0.5334     0.4704     0.8457         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.32it/s]
                   all        100        488      0.894      0.816      0.909      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   178/1000      2.93G      0.574      0.496     0.8711         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.923      0.803        0.9      0.772

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   179/1000      2.93G     0.5769      0.508     0.8558         35        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.886      0.803      0.899      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   180/1000      2.93G     0.5321     0.4596     0.8655         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.899      0.795      0.894      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   181/1000      2.93G     0.5865     0.4801      0.874         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.71it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.32it/s]
                   all        100        488      0.884      0.805      0.894      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   182/1000      2.93G       0.52     0.4568     0.8556         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.827      0.819       0.88      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   183/1000      2.93G     0.5377     0.4685     0.8721         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.873      0.775       0.87      0.748

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   184/1000      2.93G     0.5816     0.5061     0.8928         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.45it/s]
                   all        100        488      0.856      0.806       0.88      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   185/1000      2.93G      0.529     0.4603     0.8564         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.31it/s]
                   all        100        488      0.851      0.827      0.893      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   186/1000      2.93G     0.5337     0.5069     0.8552         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.894      0.813      0.913      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   187/1000      2.93G     0.5706     0.5201     0.8937         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.873      0.842      0.915      0.783

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   188/1000      2.93G     0.5795     0.5273      0.864         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.885      0.839      0.906      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   189/1000      2.93G     0.5565     0.4996      0.873         35        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.862      0.827      0.907      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   190/1000      2.93G     0.5581     0.4792     0.8638         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.847       0.72      0.881      0.739

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   191/1000      2.93G     0.5252     0.4748     0.8586         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.804      0.816      0.895      0.765

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   192/1000      2.93G     0.5132     0.4791     0.8592         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.886      0.784        0.9       0.77

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   193/1000      2.93G     0.5413     0.4712     0.8557         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.877      0.809      0.901      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   194/1000      2.93G     0.5531     0.4974     0.8522         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.13it/s]
                   all        100        488      0.855      0.825      0.903      0.784

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   195/1000      2.93G     0.5373     0.4892     0.8608         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.857      0.802      0.899      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   196/1000      2.93G     0.5459     0.4971     0.8729         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488       0.83      0.827      0.898      0.773

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   197/1000      2.93G     0.5365     0.4874     0.8604         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488       0.87      0.789      0.898      0.767

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   198/1000      2.93G     0.5519     0.4946      0.866         39        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488       0.88      0.822      0.914       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   199/1000      2.93G     0.5647     0.4898     0.8765         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.896      0.822      0.916      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   200/1000      2.93G     0.5188     0.4681     0.8342         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.54it/s]
                   all        100        488      0.863      0.853      0.912      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   201/1000      2.93G     0.5534     0.5444      0.864          5        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.879      0.841      0.912      0.784

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   202/1000      2.93G     0.5358     0.4964     0.8569         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.879      0.794      0.905      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   203/1000      2.93G     0.5312     0.4613     0.8555         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.851       0.81      0.895      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   204/1000      2.93G     0.5562     0.4812     0.8738         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.881      0.801      0.892      0.763

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   205/1000      2.93G      0.581     0.5565     0.8735         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.877      0.795      0.883      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   206/1000      2.93G     0.5473     0.5256     0.8601         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.854      0.752      0.851      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   207/1000      2.93G     0.5241     0.4721     0.8371         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.843      0.831      0.903      0.758

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   208/1000      2.93G     0.5795     0.4916     0.8848         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.876      0.818      0.913      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   209/1000      2.93G      0.545     0.5043     0.8566         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.51it/s]
                   all        100        488      0.883      0.833      0.919      0.788

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   210/1000      2.93G     0.5411      0.495       0.84         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.18it/s]
                   all        100        488      0.886      0.801      0.904      0.755

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   211/1000      2.93G     0.5475     0.4978     0.8549         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.859      0.796      0.897      0.754

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   212/1000      2.93G     0.5615     0.5368     0.8705         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.868      0.805      0.908      0.763

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   213/1000      2.93G     0.5357     0.5328     0.8786         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.863      0.816      0.907      0.772

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   214/1000      2.93G     0.5124     0.4808       0.86         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.879      0.802      0.902      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   215/1000      2.93G     0.5183     0.4673     0.8675         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]
                   all        100        488      0.874      0.804      0.899      0.766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   216/1000      2.93G     0.4988     0.4379     0.8403         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.843      0.843      0.903      0.767

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   217/1000      2.93G     0.5395     0.4865     0.8871         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.858      0.842      0.904      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   218/1000      2.93G      0.552     0.4736     0.8575         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.851      0.847      0.905      0.775

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   219/1000      2.93G     0.5245      0.495     0.8467         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.868      0.818        0.9      0.768

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   220/1000      2.93G     0.5251     0.4561     0.8389         40        320: 100%|██████████| 7/7 [00:01<00:00,  5.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.873        0.8      0.896      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   221/1000      2.93G      0.515     0.4895     0.8354         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.883       0.81      0.901      0.775

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   222/1000      2.93G     0.5185     0.4486     0.8567         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.888      0.817      0.902      0.773

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   223/1000      2.93G      0.513     0.4749       0.86         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.851      0.815      0.894      0.781

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   224/1000      2.93G     0.5553     0.4573     0.8547         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.833      0.835      0.894      0.766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   225/1000      2.93G      0.518     0.4809     0.8543         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.833      0.836      0.892      0.768

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   226/1000      2.93G     0.5473     0.4988     0.8645         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.64it/s]
                   all        100        488      0.837       0.81      0.882      0.765

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   227/1000      2.93G     0.5349     0.4862     0.8671         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.846      0.828      0.886      0.764

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   228/1000      2.93G     0.5619     0.4901     0.8607         10        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.864      0.823      0.882      0.755

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   229/1000      2.93G     0.5199     0.4851     0.8679         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488       0.87       0.81       0.88      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   230/1000      2.93G     0.5389     0.5104     0.8528         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.814      0.849      0.876      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   231/1000      2.93G     0.5514     0.5216     0.8752         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]
                   all        100        488      0.843        0.8      0.868       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   232/1000      2.93G     0.5347     0.5082     0.8729         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.867      0.796      0.882       0.76

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   233/1000      2.93G     0.5466     0.5064     0.8637         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.881      0.789      0.888      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   234/1000      2.93G     0.5471      0.467     0.8564         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.805      0.825      0.883       0.74

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   235/1000      2.93G     0.5486     0.4751     0.8531         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488       0.89      0.833      0.904      0.766

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   236/1000      2.93G     0.5433      0.483     0.8559         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.921      0.814      0.918      0.781

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   237/1000      2.93G     0.5796     0.5007     0.8518         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.912      0.822      0.918      0.772

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   238/1000      2.93G     0.5132     0.4696     0.8483         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488       0.92      0.816      0.917      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   239/1000      2.93G      0.534     0.4763     0.8507         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.924      0.793      0.914      0.784

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   240/1000      2.93G     0.5363     0.4744      0.887         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.888      0.813       0.91      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   241/1000      2.93G     0.5366     0.5288     0.8742          9        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.873      0.831      0.911      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   242/1000      2.93G     0.5193     0.4561     0.8582         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.26it/s]
                   all        100        488      0.886      0.804      0.908      0.772

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   243/1000      2.93G     0.4947     0.4704     0.8387         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.08it/s]
                   all        100        488      0.847      0.845      0.912      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   244/1000      2.93G     0.5071     0.4461     0.8412         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.848      0.862      0.918      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   245/1000      2.93G     0.5115     0.4326     0.8572         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.849      0.864      0.915      0.788

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   246/1000      2.93G     0.5165       0.42     0.8655         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.899      0.799      0.909      0.776

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   247/1000      2.93G     0.4932     0.4473     0.8657         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.22it/s]
                   all        100        488      0.873      0.823      0.916      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   248/1000      2.93G     0.5225     0.4503     0.8553         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.45it/s]
                   all        100        488      0.832      0.854      0.917      0.779

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   249/1000      2.93G     0.5096     0.4699     0.8585         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488       0.88      0.816      0.912      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   250/1000      2.93G      0.529      0.479     0.8637         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.903      0.796      0.901      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   251/1000      2.93G     0.4724     0.4179     0.8431         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.875      0.831      0.913      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   252/1000      2.93G     0.5485     0.4545     0.8799         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.868      0.845      0.918      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   253/1000      2.93G     0.4916     0.4377     0.8471         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488       0.87      0.831      0.915      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   254/1000      2.93G      0.525     0.4581     0.8453         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.882       0.83       0.92      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   255/1000      2.93G     0.5251     0.4373     0.8449         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.893      0.832      0.922      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   256/1000      2.93G     0.4865     0.4321     0.8383         37        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.865      0.814       0.91      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   257/1000      2.93G     0.4819     0.4304     0.8281         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.868      0.808      0.906      0.785

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   258/1000      2.93G     0.5006     0.4444     0.8438         35        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.905       0.82      0.918      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   259/1000      2.93G     0.4982      0.414     0.8365         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.855      0.845      0.915      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   260/1000      2.93G     0.5013     0.4303     0.8477         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.51it/s]
                   all        100        488      0.911       0.79      0.909      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   261/1000      2.93G     0.4919     0.3955     0.8454         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.902      0.804      0.908      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   262/1000      2.93G     0.4966     0.4599     0.8705         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.865      0.844       0.91      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   263/1000      2.93G     0.5052     0.4671     0.8858         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.892      0.822      0.915      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   264/1000      2.93G     0.5405     0.4596     0.8825         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.07it/s]
                   all        100        488      0.865      0.857      0.916      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   265/1000      2.93G     0.4827     0.4106     0.8456         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.855      0.858      0.917      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   266/1000      2.93G     0.5134     0.4478     0.8696         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.882       0.82      0.908      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   267/1000      2.93G     0.5085     0.4549     0.8626         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.859      0.805      0.898       0.75

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   268/1000      2.93G     0.4939     0.4484     0.8644         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.855      0.833      0.899       0.77

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   269/1000      2.93G     0.5152     0.4584     0.8538         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.54it/s]
                   all        100        488      0.831      0.841      0.898      0.772

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   270/1000      2.93G     0.4674     0.4204     0.8397         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.878      0.787      0.894      0.735

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   271/1000      2.93G      0.525     0.5233     0.8872         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488       0.85      0.821       0.89      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   272/1000      2.93G     0.5099      0.429     0.8506         40        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.24it/s]
                   all        100        488      0.847      0.849        0.9      0.769

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   273/1000      2.93G     0.4771     0.4306     0.8369         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.71it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.53it/s]
                   all        100        488      0.854      0.849      0.902      0.775

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   274/1000      2.93G     0.4745     0.4128     0.8292         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.881      0.827      0.905      0.784

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   275/1000      2.93G     0.4932     0.4612     0.8412          9        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.848      0.872      0.909      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   276/1000      2.93G     0.4917     0.4321     0.8484         41        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.861      0.855      0.912        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   277/1000      2.93G     0.5633     0.5598     0.9328          8        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.895      0.833      0.913      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   278/1000      2.93G     0.4983     0.4179     0.8395         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.862      0.841      0.899      0.753

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   279/1000      2.93G      0.491     0.4382     0.8462         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.865      0.799      0.889      0.734

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   280/1000      2.93G     0.5121      0.446     0.8751         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.32it/s]
                   all        100        488      0.861      0.793      0.885      0.723

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   281/1000      2.93G     0.4829     0.4519     0.8429         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.866      0.816      0.888       0.72

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   282/1000      2.93G     0.5757     0.5601     0.8901         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.877      0.805      0.897      0.745

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   283/1000      2.93G     0.4939     0.4225     0.8442         46        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.844      0.835      0.897      0.756

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   284/1000      2.93G     0.5171     0.4918     0.8486         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.883      0.816      0.892      0.759

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   285/1000      2.93G      0.496     0.4373     0.8526         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.882      0.838      0.897      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   286/1000      2.93G     0.4912     0.3985      0.849         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.892      0.839      0.906      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   287/1000      2.93G     0.5093     0.4287     0.8624         38        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.54it/s]
                   all        100        488      0.899      0.843      0.916      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   288/1000      2.93G     0.4821     0.4052     0.8498         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.23it/s]
                   all        100        488       0.91      0.824      0.915      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   289/1000      2.93G     0.5117     0.4343     0.8519         39        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.891      0.816      0.912      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   290/1000      2.93G     0.4735     0.4545     0.8431         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.904      0.811       0.91      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   291/1000      2.93G      0.488     0.4368     0.8232         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.915      0.817      0.911      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   292/1000      2.93G     0.4962     0.4388     0.8627         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488       0.87      0.841      0.908      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   293/1000      2.93G     0.4974     0.4247     0.8541         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.844      0.861      0.904      0.787

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   294/1000      2.93G     0.5149     0.4183     0.8463         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.854      0.839      0.893      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   295/1000      2.93G     0.5207     0.4765     0.8708         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.874      0.811      0.892      0.764

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   296/1000      2.93G     0.4883     0.4324      0.841         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.884      0.825      0.906      0.795

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   297/1000      2.93G     0.5006     0.4464     0.8495         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.889      0.822       0.91      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   298/1000      2.93G     0.4886     0.4706     0.8488         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.893      0.825      0.912      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   299/1000      2.93G     0.4866     0.4294     0.8459         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.872       0.85      0.918      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   300/1000      2.93G      0.487     0.4092     0.8544         37        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.915       0.84      0.919      0.781

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   301/1000      2.93G     0.4935     0.4218     0.8506         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.32it/s]
                   all        100        488      0.909      0.835      0.919      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   302/1000      2.93G     0.5074     0.4724     0.8567         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.913      0.833       0.92      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   303/1000      2.93G      0.471     0.4237     0.8387         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.22it/s]
                   all        100        488      0.922      0.824      0.922      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   304/1000      2.93G     0.4846     0.4592     0.8506          9        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.933      0.828      0.931      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   305/1000      2.93G     0.4802     0.4047     0.8506         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.901      0.857      0.929      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   306/1000      2.93G      0.504     0.4557     0.8612         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.914      0.845      0.929      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   307/1000      2.93G      0.502     0.4491     0.8641         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.898      0.847      0.931      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   308/1000      2.93G     0.5055     0.4451     0.8616         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.901      0.863      0.932      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   309/1000      2.93G     0.4853     0.4174     0.8598         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.55it/s]
                   all        100        488      0.895      0.864      0.931      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   310/1000      2.93G     0.4894     0.4182     0.8498         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.914      0.836      0.927      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   311/1000      2.93G      0.487     0.4145     0.8553         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488        0.9      0.822      0.919      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   312/1000      2.93G     0.5237     0.4584     0.8519         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.879      0.821      0.918      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   313/1000      2.93G     0.4902     0.4392     0.8665         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.877       0.82      0.919      0.789

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   314/1000      2.93G     0.5067     0.4387     0.8483         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.56it/s]
                   all        100        488      0.877      0.835       0.92      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   315/1000      2.93G     0.5029     0.4593      0.831         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.872      0.843      0.922      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   316/1000      2.93G     0.4708     0.3989     0.8349         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.892      0.812      0.919      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   317/1000      2.93G     0.4885     0.4291      0.855         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.864      0.822      0.912      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   318/1000      2.93G     0.4568     0.4051     0.8291         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488       0.83      0.844      0.901      0.798

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   319/1000      2.93G     0.4716     0.4381     0.8413         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.18it/s]
                   all        100        488       0.86      0.836        0.9      0.787

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   320/1000      2.93G     0.4724       0.44      0.852         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.863      0.837      0.904      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   321/1000      2.93G      0.482     0.4312     0.8242         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.849      0.835      0.905      0.798

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   322/1000      2.93G     0.4702     0.4212     0.8459         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.905      0.812      0.905       0.79

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   323/1000      2.93G     0.4828     0.4461     0.8458         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.31it/s]
                   all        100        488      0.914      0.833      0.923      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   324/1000      2.93G     0.4841     0.3976     0.8066         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.45it/s]
                   all        100        488      0.913       0.81      0.922      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   325/1000      2.93G     0.4725     0.4333     0.8616         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.38it/s]
                   all        100        488      0.887      0.818      0.918      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   326/1000      2.93G     0.4759     0.4233     0.8542         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.45it/s]
                   all        100        488      0.876      0.853      0.918      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   327/1000      2.93G     0.4711      0.434     0.8369         49        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.872      0.863      0.913      0.787

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   328/1000      2.93G     0.4888     0.4288     0.8481         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.862      0.849      0.916      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   329/1000      2.93G      0.505     0.4438     0.8762         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.886      0.849      0.916      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   330/1000      2.93G     0.4661     0.4202     0.8177         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.904      0.815        0.9      0.771

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   331/1000      2.93G     0.4974     0.4287     0.8516         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.897       0.77      0.876      0.728

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   332/1000      2.93G     0.4775     0.4497     0.8659         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.868      0.797      0.887      0.746

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   333/1000      2.93G     0.4932     0.4254     0.8597         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488       0.88      0.809      0.899      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   334/1000      2.93G     0.4756     0.4676     0.8561         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.888      0.826      0.909      0.765

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   335/1000      2.93G     0.4834     0.4391     0.8688         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.882      0.838      0.917      0.777

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   336/1000      2.93G     0.4643     0.3933     0.8459         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.922      0.832      0.924      0.788

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   337/1000      2.93G     0.5008     0.4509     0.8816         10        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.892       0.85      0.922      0.786

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   338/1000      2.93G     0.5014     0.4184     0.8579         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.869      0.856      0.915      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   339/1000      2.93G     0.4893     0.4309     0.8465         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.901      0.836      0.915      0.778

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   340/1000      2.93G     0.5069     0.4271     0.8725         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.835      0.893      0.917      0.788

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   341/1000      2.93G     0.4948     0.4264     0.8444         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488       0.87      0.866      0.919       0.79

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   342/1000      2.93G     0.4936     0.4189     0.8615         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.53it/s]
                   all        100        488      0.859      0.875      0.916      0.787

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   343/1000      2.93G     0.4705     0.4193     0.8516         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.61it/s]
                   all        100        488      0.894      0.847      0.913       0.79

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   344/1000      2.93G     0.4614     0.4238     0.8466         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.892      0.852      0.924      0.798

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   345/1000      2.93G     0.4681       0.41     0.8256         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.53it/s]
                   all        100        488      0.857      0.841      0.919      0.793

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   346/1000      2.93G      0.449      0.406     0.8337         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.877      0.821      0.911      0.792

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   347/1000      2.93G     0.4708     0.4191     0.8457         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.877      0.805      0.907       0.79

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   348/1000      2.93G     0.4563     0.4061     0.8375         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.884       0.82      0.911       0.79

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   349/1000      2.93G     0.4965       0.45     0.8579         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.842      0.849      0.909      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   350/1000      2.93G      0.468     0.3956     0.8335         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.848      0.845      0.907        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   351/1000      2.93G      0.488      0.401     0.8388         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.55it/s]
                   all        100        488      0.888      0.808      0.904      0.798

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   352/1000      2.93G     0.4611      0.388     0.8217         43        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488       0.89      0.823      0.908      0.799

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   353/1000      2.93G     0.4677     0.3704     0.8282         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.885      0.841      0.916      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   354/1000      2.93G     0.4305     0.3531     0.8142         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.863       0.86      0.919      0.799

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   355/1000      2.93G     0.4488     0.4493     0.8344         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.888      0.787      0.899      0.752

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   356/1000      2.93G      0.492     0.4058     0.8457         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.811      0.764      0.854      0.668

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   357/1000      2.93G     0.4723     0.4675     0.8601         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.794      0.784      0.878      0.705

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   358/1000      2.93G     0.4906     0.3867     0.8405         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.32it/s]
                   all        100        488       0.85      0.818      0.906      0.783

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   359/1000      2.93G      0.472     0.4075     0.8446         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.857      0.813      0.893      0.744

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   360/1000      2.93G      0.472     0.4217     0.8452         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.849       0.82      0.886      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   361/1000      2.93G       0.45      0.394     0.8343         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.844       0.83      0.877      0.743

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   362/1000      2.93G     0.4874     0.4495     0.8563         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.854      0.793      0.884      0.737

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   363/1000      2.93G     0.4809     0.4237     0.8521         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.22it/s]
                   all        100        488      0.874      0.786      0.891      0.742

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   364/1000      2.93G     0.4902     0.4177     0.8727         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.895      0.813      0.912      0.762

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   365/1000      2.93G     0.4526     0.3758     0.8252         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.873      0.831      0.916      0.765

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   366/1000      2.93G     0.4705     0.4081     0.8376         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.878      0.827      0.916      0.781

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   367/1000      2.93G     0.4821     0.4386     0.8342         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.15it/s]
                   all        100        488      0.927      0.797      0.922      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   368/1000      2.93G     0.5003     0.4018     0.8496         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.901      0.823      0.913      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   369/1000      2.93G     0.4769     0.4117     0.8442         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.904      0.799      0.909      0.791

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   370/1000      2.93G     0.4599     0.3826     0.8295         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.885      0.838      0.912      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   371/1000      2.93G     0.4893      0.423     0.8388         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.52it/s]
                   all        100        488      0.882      0.853      0.914      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   372/1000      2.93G     0.4544     0.3783     0.8358         47        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.888      0.849      0.913      0.794

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   373/1000      2.93G     0.4845     0.4028     0.8492         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.891      0.845      0.915      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   374/1000      2.93G     0.4431     0.4041     0.8443         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.911      0.839      0.917      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   375/1000      2.93G     0.4719     0.4166      0.853         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.903      0.831      0.918      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   376/1000      2.93G     0.4411     0.3838     0.8448         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.892      0.848      0.911      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   377/1000      2.93G      0.442     0.4061     0.8472         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.896      0.842      0.908      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   378/1000      2.93G     0.4799      0.426     0.8586         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.32it/s]
                   all        100        488      0.894      0.838      0.912      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   379/1000      2.93G     0.5066     0.4142     0.8621         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.867      0.868      0.914      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   380/1000      2.93G     0.4411     0.3937     0.8322         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.913       0.84      0.918      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   381/1000      2.93G     0.4757     0.3931     0.8492         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.69it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488       0.88      0.861      0.919       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   382/1000      2.93G      0.433     0.3646     0.8291         18        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488       0.91      0.867       0.93      0.823

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   383/1000      2.93G     0.4984     0.4113      0.876         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]
                   all        100        488      0.899      0.884      0.935      0.835

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   384/1000      2.93G     0.4574     0.3762     0.8651         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.55it/s]
                   all        100        488      0.917      0.861      0.933      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   385/1000      2.93G     0.4736     0.4091     0.8448         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.937      0.849      0.935      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   386/1000      2.93G     0.4467     0.3775     0.8446         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.939       0.84      0.935      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   387/1000      2.93G     0.4528     0.3872     0.8455         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.27it/s]
                   all        100        488      0.915      0.853      0.938      0.832

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   388/1000      2.93G     0.4785     0.4048     0.8659         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.21it/s]
                   all        100        488      0.915      0.876      0.941      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   389/1000      2.93G     0.4409     0.3561     0.8371         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.915      0.864       0.94      0.834

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   390/1000      2.93G     0.4312     0.3841     0.8279         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.912      0.838      0.936      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   391/1000      2.93G     0.4437     0.3842     0.8152         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.906      0.855      0.932      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   392/1000      2.93G     0.4516     0.4149     0.8517         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.22it/s]
                   all        100        488      0.888      0.849      0.923      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   393/1000      2.93G     0.4841     0.4047      0.873         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.887      0.838      0.916       0.78

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   394/1000      2.93G     0.4417     0.3921     0.8447         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.899      0.788        0.9      0.761

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   395/1000      2.93G     0.4469     0.3731     0.8326         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.852      0.812      0.885       0.76

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   396/1000      2.93G     0.4472     0.3964     0.8352         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.886      0.803      0.888      0.752

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   397/1000      2.93G     0.4539     0.4064     0.8386         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.894      0.788      0.894       0.75

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   398/1000      2.93G     0.4819     0.3895     0.8395         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.881      0.797      0.898      0.761

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   399/1000      2.93G     0.4557     0.3776     0.8399         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.36it/s]
                   all        100        488      0.886      0.797      0.902      0.755

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   400/1000      2.93G     0.4599     0.3982     0.8345         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.881      0.792      0.897      0.738

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   401/1000      2.93G     0.4549     0.3873      0.844         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.839      0.852      0.906      0.751

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   402/1000      2.93G     0.4233     0.3916     0.8448         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488       0.91      0.856      0.934      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   403/1000      2.93G     0.4303     0.3678     0.8469         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.918      0.876      0.943      0.831

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   404/1000      2.93G     0.4205     0.3508     0.8268         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.31it/s]
                   all        100        488      0.935      0.827      0.941      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   405/1000      2.93G     0.4395     0.3924     0.8376         12        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.936      0.821      0.938      0.827

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   406/1000      2.93G     0.4415     0.3731      0.832         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.912      0.841      0.932      0.828

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   407/1000      2.93G     0.4635     0.3929     0.8314         39        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.925      0.831      0.929      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   408/1000      2.93G      0.427     0.3666     0.8232         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.902      0.839      0.926      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   409/1000      2.93G     0.4389     0.3877      0.842         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488      0.923      0.835      0.927      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   410/1000      2.93G     0.4441      0.392     0.8607         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.16it/s]
                   all        100        488      0.908      0.877      0.934      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   411/1000      2.93G     0.4473     0.3869     0.8453         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.54it/s]
                   all        100        488       0.93       0.85      0.934       0.83

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   412/1000      2.93G     0.4623     0.3997     0.8326         44        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.901      0.859      0.931      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   413/1000      2.93G     0.4162     0.3555     0.8373         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.875      0.864      0.927      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   414/1000      2.93G     0.4313      0.378     0.8359         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.917       0.84      0.928      0.822

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   415/1000      2.93G     0.4605     0.4152     0.8516         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.928      0.844      0.927      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   416/1000      2.93G     0.4588     0.4011     0.8499         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.891      0.871      0.919      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   417/1000      2.93G     0.4201     0.3934     0.8463         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.899      0.868      0.923      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   418/1000      2.93G     0.4229      0.352     0.8235         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.54it/s]
                   all        100        488      0.924       0.83      0.929      0.822

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   419/1000      2.93G      0.422      0.362     0.8237         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.918      0.818      0.927      0.828

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   420/1000      2.93G     0.4471     0.4119     0.8355         22        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.875      0.865      0.926      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   421/1000      2.93G     0.4499     0.4107      0.836         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488       0.86      0.877      0.928      0.821

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   422/1000      2.93G     0.4557     0.3971     0.8547         47        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.32it/s]
                   all        100        488      0.888      0.858      0.926      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   423/1000      2.93G     0.4299     0.3838     0.8359         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.884      0.852      0.923      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   424/1000      2.93G     0.4209     0.3925     0.8631         10        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.878      0.864      0.923      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   425/1000      2.93G      0.462     0.4029     0.8534         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.24it/s]
                   all        100        488      0.891      0.849       0.92      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   426/1000      2.93G     0.4465     0.3952     0.8303         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.868      0.864      0.916      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   427/1000      2.93G     0.4665     0.3926      0.862         38        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.48it/s]
                   all        100        488       0.88      0.848      0.907      0.799

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   428/1000      2.93G     0.4493     0.3904     0.8195         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.895       0.82      0.911      0.796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   429/1000      2.93G     0.4348     0.3664     0.8383         39        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.868      0.844       0.91      0.802

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   430/1000      2.93G     0.4876     0.3729     0.8434         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.889      0.836       0.91      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   431/1000      2.93G     0.4668     0.4134     0.8395         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.50it/s]
                   all        100        488      0.895      0.835      0.912      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   432/1000      2.93G      0.427     0.3612     0.8392         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.59it/s]
                   all        100        488      0.889      0.854      0.915      0.815

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   433/1000      2.93G     0.4417     0.3819     0.8342         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.909       0.83      0.916      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   434/1000      2.93G     0.4397     0.3554     0.8405         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.51it/s]
                   all        100        488      0.866      0.848      0.915      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   435/1000      2.93G     0.4488     0.3981     0.8418         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488      0.896      0.832      0.918      0.822

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   436/1000      2.93G     0.4307     0.4323      0.846         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.855      0.853      0.916      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   437/1000      2.93G     0.4356     0.4137     0.8368         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.25it/s]
                   all        100        488      0.858      0.849      0.916      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   438/1000      2.93G      0.446     0.3784     0.8557         42        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.55it/s]
                   all        100        488      0.895      0.824       0.92       0.81

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   439/1000      2.93G     0.4135     0.3564     0.8413         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.15it/s]
                   all        100        488      0.883      0.831      0.916      0.806

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   440/1000      2.93G     0.4382     0.3757     0.8491         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.868      0.835      0.914      0.815

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   441/1000      2.93G     0.4474     0.3615      0.839         31        320: 100%|██████████| 7/7 [00:01<00:00,  5.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.878      0.822      0.912      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   442/1000      2.93G     0.4301      0.367     0.8337         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.889       0.83      0.915      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   443/1000      2.93G      0.425     0.3639     0.8373         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.867      0.852      0.917      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   444/1000      2.93G     0.4439     0.4128      0.842         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.51it/s]
                   all        100        488      0.886      0.847      0.913      0.798

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   445/1000      2.93G     0.4448     0.3891     0.8498         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.896      0.835      0.912        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   446/1000      2.93G     0.4533     0.3992     0.8385          8        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.54it/s]
                   all        100        488      0.911      0.817      0.907      0.803

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   447/1000      2.93G     0.4522     0.4001     0.8421         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.902      0.814      0.904      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   448/1000      2.93G     0.4498     0.3688     0.8266         17        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.57it/s]
                   all        100        488      0.883      0.828      0.898      0.797

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   449/1000      2.93G     0.4473     0.4044     0.8459         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488       0.89      0.824        0.9      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   450/1000      2.93G     0.4412     0.4042     0.8525         10        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.897      0.826      0.909      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   451/1000      2.93G     0.4239     0.3596     0.8322         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.892      0.827       0.91      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   452/1000      2.93G     0.4954     0.4559     0.8446          8        320: 100%|██████████| 7/7 [00:01<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.04it/s]
                   all        100        488      0.896      0.821      0.915      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   453/1000      2.93G     0.4424     0.3667     0.8358         28        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.868      0.854      0.917      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   454/1000      2.93G     0.4525     0.3762     0.8364         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.28it/s]
                   all        100        488      0.909      0.832      0.916      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   455/1000      2.93G     0.4566     0.3926     0.8259         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.26it/s]
                   all        100        488      0.882      0.857      0.916      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   456/1000      2.93G     0.4279     0.3802     0.8566         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.896      0.862      0.919      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   457/1000      2.93G     0.4397     0.4134     0.8586         29        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.18it/s]
                   all        100        488       0.91      0.857      0.923      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   458/1000      2.93G     0.4309     0.3562     0.8371         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.06it/s]
                   all        100        488      0.909      0.862      0.927      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   459/1000      2.93G     0.4204     0.3695     0.8387         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.34it/s]
                   all        100        488      0.899      0.874      0.924      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   460/1000      2.93G     0.4486     0.4149     0.8423         21        320: 100%|██████████| 7/7 [00:01<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.53it/s]
                   all        100        488      0.917      0.854      0.922      0.826

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   461/1000      2.93G     0.4443     0.4009     0.8547         19        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.909       0.86       0.92       0.82

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   462/1000      2.93G      0.457     0.3888     0.8399         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.895      0.838      0.919      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   463/1000      2.93G     0.4272     0.3844     0.8336         30        320: 100%|██████████| 7/7 [00:01<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.862      0.869      0.916      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   464/1000      2.93G      0.469     0.3784     0.8412          9        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.856      0.864      0.912      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   465/1000      2.93G     0.4872     0.4242     0.8694         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.38it/s]
                   all        100        488      0.835       0.87      0.912      0.811

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   466/1000      2.93G     0.4648     0.3943     0.8673         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.33it/s]
                   all        100        488      0.896      0.829      0.916      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   467/1000      2.93G     0.4275     0.3708     0.8414         33        320: 100%|██████████| 7/7 [00:01<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.43it/s]
                   all        100        488      0.914       0.82       0.92      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   468/1000      2.93G     0.4513     0.3745     0.8439         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.39it/s]
                   all        100        488      0.911      0.824      0.915      0.817

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   469/1000      2.93G     0.4638      0.385     0.8781         13        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.40it/s]
                   all        100        488      0.899      0.838      0.915      0.819

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   470/1000      2.93G     0.4366      0.345     0.8359         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.902      0.847      0.912      0.818

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   471/1000      2.93G     0.4296     0.3596     0.8411         20        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.904      0.843       0.91        0.8

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   472/1000      2.93G     0.4403     0.3452     0.8244         32        320: 100%|██████████| 7/7 [00:01<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488       0.87      0.856      0.918      0.807

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   473/1000      2.93G     0.4354       0.38     0.8352         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.44it/s]
                   all        100        488      0.887      0.833      0.921      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   474/1000      2.93G     0.4256     0.3645     0.8207         24        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.37it/s]
                   all        100        488      0.905      0.831      0.924      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   475/1000      2.93G     0.4306     0.3718     0.8364         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.55it/s]
                   all        100        488      0.912      0.854       0.93      0.801

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   476/1000      2.93G      0.423     0.3536     0.8441         15        320: 100%|██████████| 7/7 [00:01<00:00,  5.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.30it/s]
                   all        100        488      0.935      0.833      0.932      0.824

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   477/1000      2.93G      0.423     0.3458     0.8268         34        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.928      0.838      0.929      0.825

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   478/1000      2.93G     0.4402     0.3867     0.8437         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.51it/s]
                   all        100        488      0.908      0.836      0.924      0.804

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   479/1000      2.93G      0.434     0.3652     0.8454         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.41it/s]
                   all        100        488      0.907      0.841      0.921      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   480/1000      2.93G      0.393     0.3388     0.8231         26        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.46it/s]
                   all        100        488      0.916      0.829      0.915      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   481/1000      2.93G     0.4005     0.3552     0.8503         11        320: 100%|██████████| 7/7 [00:01<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.914      0.831      0.912      0.805

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   482/1000      2.93G     0.4275     0.3451     0.8378         36        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.903      0.836      0.909      0.808

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   483/1000      2.93G     0.4083      0.335     0.8215         27        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.49it/s]
                   all        100        488      0.888      0.837      0.908      0.809

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   484/1000      2.93G     0.3966     0.3307     0.8229         25        320: 100%|██████████| 7/7 [00:01<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.47it/s]
                   all        100        488       0.91      0.821      0.911      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   485/1000      2.93G     0.3967     0.3589     0.8447         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.22it/s]
                   all        100        488      0.885      0.836       0.91      0.816

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   486/1000      2.93G     0.4408     0.3458     0.8267         14        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.42it/s]
                   all        100        488      0.911      0.816      0.908      0.813

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   487/1000      2.93G     0.4235     0.3497      0.849          9        320: 100%|██████████| 7/7 [00:01<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.29it/s]
                   all        100        488      0.901      0.828      0.907      0.812

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   488/1000      2.93G     0.4426     0.3807     0.8386         23        320: 100%|██████████| 7/7 [00:01<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.19it/s]
                   all        100        488      0.909      0.835       0.91      0.814

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
   489/1000      2.93G     0.4225      0.363     0.8387         16        320: 100%|██████████| 7/7 [00:01<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.35it/s]
                   all        100        488      0.911      0.837      0.912      0.812
EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 389, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

489 epochs completed in 0.357 hours.
Optimizer stripped from runs\detect\train6\weights\last.pt, 19.2MB
Optimizer stripped from runs\detect\train6\weights\best.pt, 19.2MB

Validating runs\detect\train6\weights\best.pt...
Ultralytics 8.3.155  Python-3.9.21 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce GTX 1660 Ti, 6144MiB)
YOLO11s summary (fused): 100 layers, 9,416,670 parameters, 0 gradients, 21.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  4.36it/s]
                   all        100        488      0.915      0.864       0.94      0.834
                     0         44         53      0.943      0.943      0.981      0.909
                     1         36         40          1      0.788      0.878      0.742
                     2         45         65      0.833      0.754      0.904      0.809
                     3         32         45      0.853      0.911      0.929      0.805
                     4         35         49      0.917      0.898      0.938      0.842
                     5         35         48      0.907       0.81      0.928      0.825
                     6         35         37      0.885      0.838      0.925       0.83
                     7         32         43      0.928      0.904      0.968      0.871
                     8         44         57      0.909       0.93      0.978       0.87
                     9         42         51       0.97      0.863      0.967      0.842
Speed: 0.2ms preprocess, 3.3ms inference, 0.0ms loss, 2.1ms postprocess per image
Results saved to runs\detect\train6
 Learn more at https://docs.ultralytics.com/modes/train
```
