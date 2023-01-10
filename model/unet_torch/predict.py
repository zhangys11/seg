'''
This script is revised by Dr. Zhang (oo@zju.edu.cn) based on test.py.
This script is used to process pedatric fundus images.
'''
import os, sys
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch

if __package__:
    from .model import build_unet
    from .utils import create_dir, seeding
else:
    DIR = os.path.dirname(__file__)
    if DIR not in sys.path:
        sys.path.append(DIR)
    from model import build_unet
    from utils import create_dir, seeding

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

def predict_folder(input_dir, output_dir = "results/SEH", THRESHOLD = 0.02):
    '''
    THRESHOLD : the threshold to judge whether a pixel is blood vessel or background.
        You may decrease the thershold to increase detection sensativity.
    '''

    """ Seeding """
    # seeding(42)

    """ Folders """
    create_dir(output_dir)

    """ Load dataset """
    test_x = sorted(glob(input_dir + "/*")) # "../fundus/SEH/*"

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = os.path.dirname(__file__) + "/files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    time_taken = []

    for i, x in tqdm(enumerate(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)        
        OH, OW, _ = image.shape # H, W, C
        keep_aspect_size = (round(H*OW/OH), H)
        image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > THRESHOLD
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [cv2.resize(image,keep_aspect_size), line, cv2.resize(pred_y,keep_aspect_size) * 255], axis=1
        )
        cv2.imwrite(f"results/{name}.png", cat_images)
        print(f"results/{name}.png")

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)

if __name__ == "__main__":
    predict_folder(os.path.dirname(__file__) + "/../../data/fundus/SEH", 
    output_dir = os.path.dirname(__file__) + "/results/SEH", 
    THRESHOLD = 0.02)