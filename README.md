# MNIST-ODD-GAN

MNIST-Object-Detection-Dataset-GAN

The repository is the part of a project for "Bildverarbeitung und Bildverstehen". It contains some custom implementations of the most well-known GANs to generate MNIST images.

## MNIST-ObjectDetection

Original repo: https://github.com/hukkelas/MNIST-ObjectDetection/blob/master/generate_data.py

There is one problem with downloading MNIST dataset with this script. To fix the problem the whole repo was merged into current. No forks were created to avoid useless repos.

Small dataset (with 90 images total as 30/30/30) was generated with:
```
python generate_data.py --num-test-images 30 --num-train-images 30 --num-validation-images 30 --max-digits-per-image 8 --seed 42 --imsize 320 --min-digit-size 40 --max-digit-size 120
```

Large dataset (with 300 images total as 100/100/100):
```
python generate_data.py --num-test-images 100 --num-train-images 100 --num-validation-images 100 --max-digits-per-image 8 --seed 42 --imsize 320 --min-digit-size 40 --max-digit-size 120
```

Custom datasets:
```
python generate_data.py --num-test-images 0 --num-train-images 10000 --num-validation-images 0 --max-digits-per-image 8 --seed 42 --imsize 320 --min-digit-size 40 --max-digit-size 120 --source ../mnist-keras-gans/data/wgan/
```

## Environments

Environments are really tricky to manage if the GPU support is needed. Another problem is, ultralytics (framework to train yolo models) uses pytorch as backend, but I prefer to use tf/keras for my experiments. The best solution I came up with is to use two different environments.

Install torch environment:
```
conda env create -f torch-environment.yml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Install tf environment:
```
conda env create -f tf-environment.yml
```

I would simply say: works on my machine

