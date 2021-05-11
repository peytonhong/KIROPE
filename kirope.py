# -*- coding: utf-8 -*-
"""kirope.py

Kinematics driven Robot Pose Estimation with Monocular Camera.

Hyosung Hong
"""

from PIL import Image
# import requests
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

import numpy as np
import cv2
from tqdm import tqdm


class DETRdemo(nn.Module):
    """
    KIROPE implementation.

    """
    def __init__(self, num_classes, hidden_dim=256, nheads=3,
                 num_encoder_layers=7, num_decoder_layers=7):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        # self.transformer = nn.Transformer(
        #     hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nheads)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nheads)
        transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)    # [1, 256, 25, 34]  from original shape [1, 3, 800, 1066] : feature size reduced by 1/32

        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([    
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),    # [H, W, 128]
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),    # [H, W, 128]
        ], dim=-1).flatten(0, 1).unsqueeze(1)                   # [H*W, 1, 256] == [850, 1, 256]
        
        # propagate through the transformer
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),     # [H*W, 1, 256] == [850, 1, 256]
                             self.query_pos.unsqueeze(1)).transpose(0, 1)   # [100, 1, 256]
        # h.shape == [1, 100, 256]

        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h), 
                'pred_boxes': self.linear_bbox(h).sigmoid()}

"""As you can see, DETR architecture is very simple, thanks to the representational power of the Transformer. There are two main components:
* a convolutional backbone - we use ResNet-50 in this demo
* a Transformer - we use the default PyTorch nn.Transformer
Let's construct the model with 80 COCO output classes + 1 ⦰ "no object" class and load the pretrained weights.
The weights are saved in half precision to save bandwidth without hurting model accuracy.
"""

detr = DETRdemo(num_classes=91)
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cpu', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval()

"""## Computing predictions with DETR

The pre-trained DETR model that we have just loaded has been trained on the 80 COCO classes, with class indices ranging from 1 to 90 (that's why we considered 91 classes in the model construction).
In the following cells, we define the mapping from class indices to names.
"""

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

"""DETR uses standard ImageNet normalization, and output boxes in relative image coordinates in $[x_{\text{center}}, y_{\text{center}}, w, h]$ format, where $[x_{\text{center}}, y_{\text{center}}]$ is the predicted center of the bounding box, and $w, h$ its width and height. Because the coordinates are relative to the image dimension and lies between $[0, 1]$, we convert predictions to absolute image coordinates and $[x_0, y_0, x_1, y_1]$ format for visualization purposes."""

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

"""Let's put everything together in a `detect` function:"""

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0) # [1, 3, 800, 1066] image resize from transform function

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

"""## Using DETR
To try DETRdemo model on your own image just change the URL below.
"""

# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# im = Image.open(requests.get(url, stream=True).raw)

# scores, boxes = detect(im, detr, transform)

"""Let's now visualize the model predictions"""

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

color_dict = {'person': (0, 255, 0), 'baseball glove': (255, 255, 0), 'baseball bat': (255, 0, 255), 'sports ball': (0, 255, 255)}

def result_image(cv_img, prob, boxes):
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        xmin = int(xmin); ymin = int(ymin); xmax = int(xmax); ymax = int(ymax)
        cl = p.argmax()
        name = CLASSES[cl]
        text_with_p = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        
        if name in color_dict:
            # c = (np.array(c)*255).astype(np.uint8).tolist()
            # print(xmin, ymin, xmax, ymax, c)
            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), color_dict[name], 1)            
            cv2.putText(cv_img, text_with_p, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, color_dict[name], 1)

    return cv_img


class ObjectTracker():
    def __init__(self, index, xmin, ymin, xmax, ymax):
        self.index = index          # object's unique ID
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax        
        self.find_center(self, xmin, ymin, xmax, ymax)
        
        self.x_diff = 0             # x position difference
        self.y_diff = 0             # y position difference

    def update_position(self, xmin, ymin, xmax, ymax):
        self.x_diff = x_center - self.x_center
        self.y_diff = y_center - self.y_center
        self.x_center = x_center
        self.y_center = y_center

    def find_center(self, xmin, ymin, xmax, ymax):
        self.x_center = int((xmin+xmax)/2)    # bounding box center position
        self.y_center = int((ymin+ymax)/2)    # bounding box center position        
        
    



# plot_results(im, scores, boxes)

# Video detection
video_path = './data/baseball_2nd_base.mp4'
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./data/baseball_2nd_base_result_temp.mp4', fourcc, fps, (640,480))
# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         break
if cap.isOpened():
    for i in tqdm(range(length-1)):    
        ret, im = cap.read()
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (640,480)) # [ 480, 640, 3]        
        pil_img = Image.fromarray(im) # [640, 480]

        scores, boxes = detect(pil_img, detr, transform)

        result_img = result_image(im, scores, boxes)
        out.write(result_img)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
