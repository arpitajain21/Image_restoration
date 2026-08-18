import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from nafnet_v5_gan import NAFNet_Generator


class NPYFolderDataset(Dataset):
    def __init__(self, input_dir):
        self.files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)
        arr = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        return arr, os.path.basename(self.files[idx])


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with the trained NAFNet restoration model.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='Folder containing degraded .npy images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write restored .npy outputs')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = {'base_ch': 32, 'stage_blocks': 3, 'mid_blocks': 6}
    model = NAFNet_Generator(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    dataset = NPYFolderDataset(args.input_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch, names in loader:
            batch = batch.to(device)
            pred = model(batch).clamp(0, 1)
            for out, name in zip(pred.cpu().numpy(), names):
                np.save(os.path.join(args.output_dir, name), out[0])

    print(f'Inference complete. Outputs saved to {args.output_dir}')


if __name__ == '__main__':
    main()
