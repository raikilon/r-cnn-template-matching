from detection import utils
from template_dataset import TemplateDataset
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from detection import transforms
from PIL import Image

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
# Train transform
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5)
])
# Test transform
test_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = TemplateDataset("dataset", test_transform)
torch.manual_seed(1)

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# move model to the right device
model.to(device)

# pick one image from the test set
img, _ = dataset[0]
# put the model in evaluation mode
#model = torch.nn.DataParallel(model)
checkpoint = torch.load("model.pth", map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
img.show()
print(prediction)
for i in range(len(prediction[0]['masks'])):
    pred = Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy())
    pred.show()