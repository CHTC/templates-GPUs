
import os
from pathlib import Path

import torch
import typer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from train import CatAndDogConvNet, transform

from PIL import Image
class TestCatDogDataset(Dataset):
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
        fileid = path.split('/')[-1].split('.')[0]
        return (image, fileid)

def main(
    data_dir: Path = typer.Option("./data", "--data-dir", "-d", help="Directory containing the training data"),
    model_path: Path = typer.Option("./model.pth", "--model-path", "-m", help="Path to the model checkpoint file"),
):
    # Set device to cuda if available, mps if available, cpu otherwise
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # load model and prepare for inference
    model = CatAndDogConvNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # load data
    input_files = os.listdir(data_dir)
    def test_path(p): return str(data_dir / p)
    input_files = list(map(test_path, input_files))
    print(f"Checking dogginess of {len(input_files)} images.")

    test_ds = TestCatDogDataset(input_files, transform)
    test_dl = DataLoader(test_ds, batch_size=100)

    dog_probs = []

    with torch.no_grad():
        for X, fileid in test_dl:
            X= X.to(device)
            preds = model(X).to(device)
            preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
            dog_probs += list(zip(list(fileid), preds_list))

    with open('dog_probs.csv', 'w') as f:
        f.write('fileid,prob\n')
        for fileid, prob in dog_probs:
            f.write(f'{fileid},{prob}\n')

    print(f"Wrote dog_probs.csv")

if __name__ == "__main__":
    typer.run(main)