import numpy as np
import cv2
import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from network.twonet import Dual_net
import torch.nn.functional as F
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")
from network.unetpp import NestedUNet
import yaml
from lib.config import parse_args
import pdb
import time

transform = transforms.Compose([
    transforms.ToTensor(),
])


def do_overlap(data, model): # , stride=0, roi_h=512, roi_w=512
    # pdb.set_trace()
    _, _, w, h = data.shape
    # pdb.set_trace()
    # assert w==512 and h==512
    # assert roi_h == roi_w
    # assert (h - roi_h) % stride == 0
    output = torch.zeros(1, 1, w, h)
    frequency = torch.zeros(1, 1, w, h)

    # number = 1 # number = int((h - roi_h) / stride + 1) # 1

    # predict
    # pdb.set_trace()
    pred = model(data, ratio=[1])[0]
    infer_time = model(data, ratio=[1])[-1]
    pred = F.softmax(pred, dim=1)
    pred = pred[0, 1, ...].cpu()

    # pred = weight_mul(pred)


    # output[output > 0.5] = 1
    # output[output <= 0.5] = 0

    return pred, infer_time


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_overlay(im1,im2):
    dst = Image.new('RGB', (im1.width, im1.height))
    dst.paste(im1, (0,0))
    dst.paste(im2, (0,0), mask = im2)
    return dst

def infer(model, device, cfg):
    data_path=cfg['TEST_DATA_PATH']
    prediction_path=cfg['TEST_PRED_PATH']

    file_name = sorted(os.listdir(data_path))
    # pdb.set_trace()
    with torch.no_grad():
        time = []
        for i in range(len(file_name)):
            img = Image.open(data_path + "/" + file_name[i]).convert('RGB')

            # resize to the training stage size
            W, H = img.size
            W = round(584/H * W)
            H = 584
            img = img.resize((W,H))
            
            if cfg["INPUT_CHANNEL"]==1:
                data = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255
            elif cfg["INPUT_CHANNEL"]==3:
                data = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255
            else:
                raise RuntimeError('Please check input channel of the dataset.')
            # data = preprocess(data)

            data = data.to(device).unsqueeze(0)

            pred, infer_time = do_overlap(data, model)
            time.append(infer_time)
            pred = pred.cpu().numpy()

            if cfg['THRESHOLD'] > 0:
                pred = pred > cfg['THRESHOLD']

            pred = pred * 255
            # pdb.set_trace()
            pred = Image.fromarray(np.uint8(pred))

            if cfg['COMBINE']:
                combined = get_concat_h(img, pred)
                overlay = get_overlay(img, pred)
                combined = get_concat_h(combined, overlay)
            else:
                combined = pred

            combined.save(prediction_path + "/" + str(i + 1) + "_" + file_name[i])
            # pdb.set_trace()

    time = np.array(time)
    print("time", time.mean())

if __name__ == "__main__":
    args = parse_args()
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    model_num = cfg['MODEL_NUMBER']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Dual_net(cfg).cuda()
    # pdb.set_trace()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
        model.to(device)

    model.load_state_dict(torch.load(cfg['MODEL_PATH'] + "/" + "{}.pth".format(model_num)))
    model.eval()

    if not os.path.isdir(cfg['TEST_PRED_PATH']):
        os.makedirs(cfg['TEST_PRED_PATH'])

    infer(model=model,
          device=device,
          cfg=cfg)
