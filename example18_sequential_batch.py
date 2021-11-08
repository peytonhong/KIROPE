import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from dataset_load import RobotDataset
from kirope_model import ResnetSimple
# from utils.digital_twin import DigitalTwin

def str2bool(v):
    # Converts True or False for argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def argparse_args():
  desc = "Pytorch implementation of 'Kinematics Guided Robot Pose Estimation with Monocular Camera (KIROPE)'"
  parser = argparse.ArgumentParser(description=desc)
#   parser.add_argument('command', help="'train' or 'evaluate'")
  parser.add_argument('--num_epochs', default=100, type=int, help="The number of epochs to run")
  parser.add_argument('--batch_size', default=3, type=int, help="The number of batchs for each epoch")
  parser.add_argument('--digital_twin', default=False, type=str2bool, help="True: run digital twin simulation on testing")
  parser.add_argument('--aug', default=True, type=str2bool, help="True: image and keypoint augmentation")
  parser.add_argument('--resume', default=False, type=str2bool, help="True: Load the trained model and resume training")  
  
  return parser.parse_args()


def main(args):

    lr = 1e-3           # learning rate
    model_path = './checkpoints/model_best.pth.tar'
    model = ResnetSimple(num_joints=6)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if torch.cuda.is_available(): # for multi gpu compatibility
        device = 'cuda'        
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).to(device)
    else:
        device = 'cpu'

    train_dataset = RobotDataset(data_dir='annotation/real/test')
    train_seq_sampler = torch.utils.data.SequentialSampler(train_dataset)
    train_sampler = torch.utils.data.BatchSampler(train_seq_sampler, batch_size=args.batch_size, drop_last=True)
    train_subset = torch.utils.data.Subset(train_dataset, [[10,13,16],[11,14,17],[12,15,18]])
    train_iterator = DataLoader(dataset=train_subset, batch_size=args.batch_size, shuffle=False, sampler=None)

    train(args, model, train_iterator, device, optimizer)


def train(args, model, dataset_iterator, device, optimizer):
    
    for iter, sampled_batch in enumerate(dataset_iterator):
        print(iter, sampled_batch['image_path_1'])
        # image = sampled_batch['image_1'] # tensor [N, 3, 480, 640]
        # gt_belief_maps = sampled_batch['belief_maps_1']
        # image, gt_belief_maps = image.to(device), gt_belief_maps.to(device)
        # optimizer.zero_grad()
        # output = model(image) # ResNet model
        # loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)
        # loss.backward()
        # optimizer.step()

        if iter>2:
            exit()

        

    


if __name__ == '__main__':
    # parse arguments
    args = argparse_args()
    if args is None:
        exit()
    print(args)

    main(args)