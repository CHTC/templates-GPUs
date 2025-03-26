"""
https://www.doczamora.com/cats-vs-dogs-binary-classifier-with-pytorch-cnn
"""

import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import typer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image

# image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Pytorch Convolutional Neural Network Model Architecture
class CatAndDogConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        # connected layers
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)


    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)

        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X

# preprocessing of images
class CatDogDataset(Dataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.transform = transform

    def __len__(self): return self.len

    def __getitem__(self, index): 
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        label = 0 if 'cat' in path else 1
        return (image, label)

def main(
    data_dir: Path = typer.Option("./data", "--data-dir", "-d", help="Directory containing the training data"),
    checkpoint_dir: Path = typer.Option("./checkpoints", "--checkpoint-dir", "-c", help="Directory to save model checkpoints"),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
):
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.pth"
    
    img_files = os.listdir(data_dir / "train")
    img_files = list(filter(lambda x: x != 'train', img_files))
    def train_path(p): return str(data_dir / "train" / p)
    img_files = list(map(train_path, img_files))

    print("total training images", len(img_files))
    print("First item", img_files[0])

    # create train-test split
    random.shuffle(img_files)

    train = img_files[:20000]
    test = img_files[20000:]

    print("train size", len(train))
    print("test size", len(test))

    # Set device to cuda if available, mps if available, cpu otherwise
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # create train dataset
    train_ds = CatDogDataset(train, transform)
    train_dl = DataLoader(train_ds, batch_size=500)

    # create test dataset
    test_ds = CatDogDataset(test, transform)
    test_dl = DataLoader(test_ds, batch_size=500)

    # Create instance of the model and move to device
    model = CatAndDogConvNet().to(device)

    losses = []
    accuracies = []
    start = time.time()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    if not os.path.exists(model_path):
        # Model Training...
        start = time.time()  # start timer for training loop

        for epoch in range(epochs):

            epoch_loss = 0
            epoch_accuracy = 0

            for X, y in train_dl:
                # Move data to device
                X, y = X.to(device), y.to(device)
                
                preds = model(X)
                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = ((preds.argmax(dim=1) == y).float().mean())
                epoch_accuracy += accuracy
                epoch_loss += loss
                print('.', end='', flush=True)

            epoch_accuracy = epoch_accuracy/len(train_dl)
            accuracies.append(epoch_accuracy)
            epoch_loss = epoch_loss / len(train_dl)
            losses.append(epoch_loss)

            print("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))

            # test set accuracy
            with torch.no_grad():

                test_epoch_loss = 0
                test_epoch_accuracy = 0

                for test_X, test_y in test_dl:
                    # Move test data to device
                    test_X, test_y = test_X.to(device), test_y.to(device)

                    test_preds = model(test_X)
                    test_loss = loss_fn(test_preds, test_y)

                    test_epoch_loss += test_loss            
                    test_accuracy = ((test_preds.argmax(dim=1) == test_y).float().mean())
                    test_epoch_accuracy += test_accuracy

                test_epoch_accuracy = test_epoch_accuracy/len(test_dl)
                test_epoch_loss = test_epoch_loss / len(test_dl)

                print("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch, test_epoch_loss, test_epoch_accuracy, time.time() - start))

        # save model
        torch.save(model.state_dict(), model_path)
        # stop timer and report time spent
        end = time.time()
        print(f"Model saved to {model_path}. Training completed in {(end-start):.2f} seconds on {device}")
    else:
        model = CatAndDogConvNet()
        model.to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

    # if test:
    #     test_files = os.listdir(data_dir / "test")
    #     test_files = list(filter(lambda x: x != 'test', test_files))
    #     def test_path(p): return str(data_dir / "test" / p)
    #     test_files = list(map(test_path, test_files))


    #     test_ds = TestCatDogDataset(test_files, transform)
    #     test_dl = DataLoader(test_ds, batch_size=100)
    #     len(test_ds), len(test_dl)


    #     dog_probs = []

    #     with torch.no_grad():
    #         for X, fileid in test_dl:
    #             X= X.to(device)
    #             preds = model(X).to(device)
    #             preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
    #             dog_probs += list(zip(list(fileid), preds_list))

    #     # display some images
    #     for img, probs in zip(test_files[:10], dog_probs[:10]):
    #         pil_im = Image.open(img, 'r')
    #         label = "dog" if probs[1] > 0.5 else "cat"
    #         title = "prob of dog: " + str(probs[1]) + " Classified as: " + label
    #         plt.figure()
    #         plt.imshow(pil_im)
    #         plt.suptitle(title)
    #         plt.show()

if __name__ == "__main__":
    typer.run(main)