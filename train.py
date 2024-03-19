import os
import time
import datetime
import numpy as np
import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from utils import seeding, create_dir, print_and_save, shuffling, epoch_time, calculate_metrics
# from model import U_Net
# from  model import NestedUNet
from model import TResUnet
from utils.metrics import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def load_names(path, file_path,flag:str):
    f = open(file_path, "r")
    data = f.read().split("\n")[:-1]
    images = [os.path.join(path,flag,"images", name).replace('\\','/')  for name in data]
    masks = [os.path.join(path,flag,"masks", name).replace('\\','/')  for name in data]
    return images, masks

def load_data(path,flag:str):
    train_names_path = f"{path}/train.txt"
    valid_names_path = f"{path}/val.txt"
    test_names_path = f"{path}/test.txt"
    if flag=='train':
        train_x, train_y = load_names(path, train_names_path, 'train')
        valid_x, valid_y = load_names(path, valid_names_path, 'val')
        return (train_x, train_y), (valid_x, valid_y)
    else:
        test_x, test_y = load_names(path, test_names_path, 'test')
        return (test_x, test_y)




class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)
        self.size = size

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        # image = Image.open(self.images_path[index]).convert("RGB")
        # mask = Image.open(self.masks_path[index]).convert("L")

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, (256,256))
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        mask = cv2.resize(mask, (256,256))
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        return image, mask

    def __len__(self):
        return self.n_samples

#训练过程
def train(model, loader, optimizer, loss_fn, device):
    model.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    train_bar = tqdm(enumerate(loader), desc="Train",total=len(loader))
    for i, (x, y) in train_bar:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        # y_pred1,y_pred2 = model(x)
        # loss1 = loss_fn(y_pred1, y)
        # loss2 = loss_fn(y_pred2, y)
        # loss =  loss1+loss2
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, y_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]
#验证过程
def evaluate(model, loader, loss_fn, device):
    model.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0
    var_bar = tqdm(enumerate(loader), desc="val",total=len(loader))
    with torch.no_grad():
        for i, (x, y) in var_bar:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            # y_pred1,y_pred2 = model(x)
            # loss1 = loss_fn(y_pred1, y)
            # loss2 = loss_fn(y_pred2, y)
            # loss = loss1+loss2
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, y_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    model_name = 'TResUnet'
    dataset_name = 'ISIC2018'

    writer = SummaryWriter(f"logs/{model_name}/{dataset_name}")
    """ Directories """
    create_dir(f"files/{model_name}/{dataset_name}")

    """ Training logfile """
    train_log_path = f"files/{model_name}/{dataset_name}/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open(f"files/{model_name}/{dataset_name}/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 100
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = f"files/{model_name}/{dataset_name}/checkpoint.pth"



    #训练数据集目录
    path = f"./Data/{dataset_name}/TrainDataset/"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y) = load_data(path,'train')
    train_x, train_y = shuffling(train_x, train_y)
    # train_x = train_x[:100]
    # train_y = train_y[:100]
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)



    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )

    """ Model """
    device = torch.device('cuda')
    model = TResUnet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceLoss()
    loss_name = "DiceLoss"
    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        """train"""
        train_loss, train_metrics = train(model, train_loader, optimizer, loss_fn, device)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar('Train/Jaccard', train_metrics[0], epoch)
        writer.add_scalar('Train/F1', train_metrics[1], epoch)
        writer.add_scalar('Train/Recall', train_metrics[2], epoch)
        writer.add_scalar('Train/Precision', train_metrics[3], epoch)
        """val"""
        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device)
        writer.add_scalar("Valid/Loss", valid_loss, epoch)
        writer.add_scalar('Valid/Jaccard', valid_metrics[0], epoch)
        writer.add_scalar('Valid/F1', valid_metrics[1], epoch)
        writer.add_scalar('Valid/Recall', valid_metrics[2], epoch)
        writer.add_scalar('Valid/Precision', valid_metrics[3], epoch)
        scheduler.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break