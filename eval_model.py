from detection import utils
from template_dataset import TemplateDataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from detection import transforms
from PIL import Image
import numpy as np
import cv2
import time

start = time.time()
# Finetuning Mask-R-CNN
num_classes = 4  # counting the background
# load an instance segmentation model pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   num_classes)

# Test transform
test_transform = transforms.Compose([
    transforms.ToTensor()
])
# create dataset and set seed to recreate results
dataset = TemplateDataset("dataset", test_transform)
torch.manual_seed(1)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# pick one image from the test set
img, _ = dataset[0]

# put the model in evaluation mode
# model = torch.nn.DataParallel(model)
checkpoint = torch.load("models/model.pth", map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

# transform loaded image for showing (conventional format)
img = img.mul(255).permute(1, 2, 0).byte().numpy().astype('float16')
# set values between 0 and 1 for alpha blending
img /= 255.0
# create colored layer for each template
green = np.ones(img.shape, dtype=np.float) * (0, 1, 0)
red = np.ones(img.shape, dtype=np.float) * (1, 0, 0)
blue = np.ones(img.shape, dtype=np.float) * (0, 0, 1)
# set transparency to 50%
transparency = .5

# go over all the masks and use them if the confidence of belonging to a class is above 0.5
for i in range(len(prediction[0]['masks'])):
    if prediction[0]['scores'][i].cpu().numpy() > 0.5:
        # format mask and put it between 0 and 1 for alpha blending
        mask = prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
        mask = np.array(mask, dtype=np.float)
        mask /= 255.0

        # set transparency
        mask *= transparency
        # add additional dimension for multiplication (from w x h to w x h x 1)
        mask = np.expand_dims(mask, axis=2)
        # based on the template use different color with alpha blending
        if prediction[0]['labels'][i].cpu().numpy() == 1:
            img = red * mask + img * (1.0 - mask)
        if prediction[0]['labels'][i].cpu().numpy() == 2:
            img = blue * mask + img * (1.0 - mask)
        if prediction[0]['labels'][i].cpu().numpy() == 3:
            img = green * mask + img * (1.0 - mask)

        # cv2.imshow('mask', out)
        # v2.waitKey()

# save image
cv2.imwrite("out.png", cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
end = time.time()
print(end - start)