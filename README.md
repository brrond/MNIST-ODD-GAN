# MNIST-ODD-GAN

MNIST-Object-Detection-Dataset-GAN

## MNIST-ObjectDetection

Original repo: https://github.com/hukkelas/MNIST-ObjectDetection/blob/master/generate_data.py

There is one problem with downloading MNIST dataset with this script. To fix the problem the whole repo was merged into current. No forks were created to avoid useless repos.

Command to generated the dataset:
```
python generate_data.py --num-test-images 100 --num-train-images 100 --max-digits-per-image 8
```

Install torch environment:
```
conda env create -f environment.yml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```


