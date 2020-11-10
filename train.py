import torch
import torch.nn as nn
import argparse
import numpy as np
import joblib
from time import time
import albumentations as albu
from tqdm import tqdm

from model import ViT
from data import get_loader
from utils import make_reproducible, mkdir

# for reproduciblity
make_reproducible()

# load config
parser = argparse.ArgumentParser()
parser.add_argument('--logdir', default='baseline')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=int, default=0.1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--weight_decay', type=int, default=0.0001)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--mlp_dim', type=int, default=512)
parser.add_argument('--depth', type=int, default=3)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('--dropout', type=int, default=0.1)
parser.add_argument('--emb_dropout', type=int, default=0.1)
parser.add_argument('--amp', type=int, default=0)
args = parser.parse_args()
args.amp = bool(args.amp)

# define device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# create logdir
save_path = f'logs/{args.logdir}'
mkdir(save_path)

# define augmentation
train_transform = albu.Compose([
    albu.ShiftScaleRotate(shift_limit=0.125, border_mode=0, value=0, p=1),
    albu.HorizontalFlip(p=0.5),
    albu.CoarseDropout(max_holes=3, max_height=8, max_width=8),
    albu.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
val_transform = albu.Compose([
    albu.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

# load data
x_train = joblib.load('data/x_data.jl')
y_train = joblib.load('data/y_data.jl')
x_val = joblib.load('data/x_test.jl')
y_val = joblib.load('data/y_test.jl')
train_loader = get_loader(x_train, y_train, batch_size=args.batch_size, num_workers=args.num_workers,
                          transforms=train_transform, shuffle=True)
val_loader = get_loader(x_val, y_val, batch_size=args.batch_size, num_workers=args.num_workers,
                        transforms=val_transform, shuffle=False)

print(f'# Train Samples: {len(x_train)} | # Val Samples: {len(x_val)}')

# define model
model = ViT(
    image_size=32,
    patch_size=2,
    num_classes=10,
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    dropout=args.dropout,
    emb_dropout=args.emb_dropout
).to(device)
print(f'# Parameters: {sum(p.numel() for p in model.parameters())}')

# define loss
criterion = nn.CrossEntropyLoss()

# define optimizer
kwargs = model.parameters()
optimizer = torch.optim.SGD(kwargs, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

# define lr scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.01, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                epochs=args.epochs)

# mixed precision
amp_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

best_val_metric = 0.0

# loop
t0 = time()

for epoch in range(args.epochs):

    t00 = time()

    # check current learning rates
    current_lrs = [x["lr"] for x in optimizer.param_groups]
    print(f'EPOCH: {epoch} | LRs: {set(current_lrs)}')

    # train
    model.train()
    train_loss = 0.0
    train_metric = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        # mixed precision
        amp_scaler.scale(loss).backward()
        amp_scaler.step(optimizer)
        amp_scaler.update()

        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * len(inputs) / len(x_train)
        train_metric -= (logits.detach().cpu().numpy().argmax(axis=1) == targets.detach().cpu().numpy()).mean() * len(
            inputs) / len(x_train)

    str_train_loss = np.round(train_loss, 6)
    str_train_metric = np.round(train_metric, 6)
    print(f'(trn) LOSS: {str_train_loss} | METRIC: {str_train_metric}')

    # validate
    model.eval()
    val_loss = 0.0
    val_metric = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            val_preds += logits.cpu().tolist()
            val_targets += targets.cpu().tolist()
            val_loss += loss.item() * len(inputs) / len(x_val)

    val_preds, val_targets = np.array(val_preds), np.array(val_targets)
    val_metric = -(val_preds.argmax(axis=1) == val_targets).mean()  # negative accuracy
    str_val_metric = np.round(val_metric, 6)
    str_val_loss = np.round(val_loss, 6)
    print(f'(val) LOSS: {str_val_loss} | METRIC: {str_val_metric}')

    # add log
    with open(f'{save_path}/log.txt', 'a') as f:
        f.write(
            f'{epoch} - {str_train_loss} - {str_train_metric} - {str_val_loss} - {str_val_metric} - {set(current_lrs)}\n')

    if val_metric < best_val_metric:
        # save info when improved
        best_info = {
            'epoch': epoch,
            'model': model.state_dict(),
            'amp_scaler': amp_scaler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'learning_rates': current_lrs,
            'train_loss': train_loss,
            'train_metric': train_metric,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'val_preds': val_preds,
        }
        best_val_metric = val_metric
        torch.save(best_info, f'{save_path}/best.pth')
    if epoch < 2:
        print(f'Runtime: {int(time() - t00)}')

print(f'Best Val Score: {best_val_metric}')
print(f'Runtime: {int(time() - t0)}')
