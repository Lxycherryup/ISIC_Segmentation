import os,time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from operator import add
import numpy as np
import cv2
import torch
from tqdm import tqdm
from model import U_Net
from model import TResUnet
from utils import create_dir, seeding
from utils import calculate_metrics
from train import load_data
""" Seeding """
seeding(42)

""" Load the checkpoint """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'TResUnet'
model = TResUnet()
model = model.to(device)

checkpoint_path = r"E:\ISIC_Segmentation\files\TResUnet\ISIC2018\checkpoint.pth"

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

""" Image path """
image_path = "path_to_single_image"  # Replace with the actual path to the image you want to predict
save_path = "path_to_save_results"  # Replace with the path where you want to save the results



def predict_single_image(image_path,mask_path):
    """ Predict a single image and save the result """
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    name = os.path.basename(image_path).split(".")[0]

    """ Image """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    save_img = image
    image = np.transpose(image, (2, 0, 1))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.to(device)

    with torch.no_grad():
        """ FPS calculation """
        start_time = time.time()
        heatmap, y_pred = model(image, heatmap=True)
        y_pred = torch.sigmoid(y_pred)
        end_time = time.time() - start_time

        """ Evaluation metrics """
        mask_path = mask_path
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        save_mask = mask
        save_mask = np.expand_dims(save_mask, axis=-1)
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        """ Evaluation metrics """
        score = calculate_metrics(mask, y_pred)
        metrics_score = list(map(add, metrics_score, score))

        """ Predicted Mask """
        y_pred = y_pred[0].cpu().numpy()
        y_pred = np.squeeze(y_pred, axis=0)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)
        y_pred = y_pred * 255
        y_pred = np.array(y_pred, dtype=np.uint8)
        y_pred = np.expand_dims(y_pred, axis=-1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)



    jaccard = metrics_score[0]
    f1 = metrics_score[1]
    recall = metrics_score[2]
    precision = metrics_score[3]
    acc = metrics_score[4]
    f2 = metrics_score[5]

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")
    print("Time taken: ", end_time)
    cv2.imwrite(image_path.replace('image','pred'), y_pred)
    metrics = metrics_score

    return metrics







