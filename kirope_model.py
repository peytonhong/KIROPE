from torchvision.models import resnet50, resnet101
from torch import nn
import torch.nn.functional as F


class ResnetSimple(nn.Module):
    def __init__(self, n_keypoints=7, pretrained=True):
        super(ResnetSimple, self).__init__()
        net = resnet50(pretrained=pretrained)
        # self.conv1 = net.conv1
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4


        # upconvolution and final layer
        BN_MOMENTUM = 0.1

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=2048,
                out_channels=256,
                kernel_size=5,
                stride=5,
                padding=0,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.upsample(x)


        return {'pred_belief_maps': x}






class KIROPE_Transformer(nn.Module):
    """
    KIROPE implementation.

    """
    def __init__(self, num_joints, hidden_dim=256, nheads=2,
                 num_encoder_layers=7, num_decoder_layers=7):
        super().__init__()

        self.num_joints = num_joints

        # create ResNet-50 backbone
        self.backbone = resnet50(pretrained=True)
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        # self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nheads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, nheads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.fc_out = nn.Linear(in_features=hidden_dim, out_features=25*25) # [1, 7, 625] -> after reshape: [1, 7, 25, 25]
        # self.dconv1 = nn.ConvTranspose2d(in_channels=self.num_joints, out_channels=64, kernel_size=5, stride=5, padding=0, output_padding=0) # [1, 64, 125, 125] (x5)
        # self.dconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1) # [1, 32, 250, 250] (x2)
        # self.dconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=7, kernel_size=3, stride=2, padding=1, output_padding=1) # [1, 7, 500, 500] (x2)
        # upconvolution and final layer
        BN_MOMENTUM = 0.1

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=7,
                out_channels=256,
                kernel_size=5,
                stride=5,
                padding=0,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
            ),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_joints, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, images, keypoint_embeddings):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)    # [1, 256, 25, 25]  from original shape [1, 3, 800, 800] : feature size reduced by 1/32
        
        # propagate through the transformer
        # h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),     # [H*W, 1, 256] == [850, 1, 256]
        #                      self.query_pos.unsqueeze(1)).transpose(0, 1)   # [100, 1, 256]
        # h.shape == [1, 100, 256]
        # Transformer 
        # h = self.transformer(h.flatten(2).permute(2, 0, 1), keypoint_embeddings.transpose(0,1)) # [input, embedding]
        
        h = h.flatten(2).permute(2, 0, 1)        
        h = self.transformer_encoder(h)  # [625, 1, 256]
        x = self.transformer_decoder(keypoint_embeddings.transpose(0,1), h) # [7, 1, 256]
        # print(x.shape)
        x = x.transpose(0, 1) # [1, 7, 256]
        
        x = self.fc_out(x) # [1, 7, 25*25]
        x = x.reshape(-1, self.num_joints, 25, 25)
        # x = F.relu(self.dconv1(x))
        # x = F.relu(self.dconv2(x))
        # x = self.dconv3(x)
        x = self.upsample(x)
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_belief_maps': x}