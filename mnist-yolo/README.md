# MNIST Object Detection with YOLO

For the purpose of verifying generated dataset some base models for object detection must be trained first.

train yolo model:
```
yolo train model=yolo11s.pt data=mnist.yaml epochs=1000 patience=100 batch=16 imgsz=320 save=True optimizer=adam seed=42 deterministic=True cos_lr=True amp=False lr0=0.005 lrf=0.01 momentum=0.937 warmup_epochs=5 dropout=0.3 plots=True
```

## Trained models

Following models were trained:

| Path | Description | mAP@0.5 |
| ---- | ----------- | ------- |
| runs/detect/train | yolo11s pretrained | 0.911 |
| runs/detect/train2 | yolo11s fresh | 0.883 |
| runs/detect/train3 | yolo11s pretrained | 0.0051 |
| runs/detect/train4 | yolo11s | skipped |

