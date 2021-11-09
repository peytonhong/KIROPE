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
from tqdm import tqdm
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

    train_dataset = RobotDataset(data_dir='annotation/real/train')
    
    for epoch in range(10):
        train_loss = train(args, model, train_dataset, device, optimizer)
        print(epoch, train_loss)


def train(args, model, train_dataset, device, optimizer):
    sequence_length = 10
    dir_numbers = np.random.choice(len(train_dataset.sub_dirs), args.batch_size)
    
    subset_list = []
    for i in dir_numbers:
        sub_dir_range = range(train_dataset.sub_dir_beginning_index[i], 
                              train_dataset.sub_dir_beginning_index[i] + train_dataset.num_sub_dir_files[i] - sequence_length + 1)
        beginning_index = np.random.choice(sub_dir_range)
        subset_list.append(np.arange(beginning_index, beginning_index+sequence_length))
    subset_list = np.array(subset_list).transpose().reshape(-1)
    dataset_subset = torch.utils.data.Subset(train_dataset, subset_list)
    train_iterator = DataLoader(dataset=dataset_subset, batch_size=args.batch_size, shuffle=False, sampler=None)

    train_loss_sum = 0
    num_trained_data = 0
    for iter, sampled_batch in enumerate(tqdm(train_iterator)):
        # print(iter, sampled_batch['image_path_1'])
        image = sampled_batch['image_1'] # tensor [N, 3, 480, 640]
        gt_belief_maps = sampled_batch['belief_maps_1']
        image, gt_belief_maps = image.to(device), gt_belief_maps.to(device)
        optimizer.zero_grad()
        output = model(image) # ResNet model
        loss = F.mse_loss(output['pred_belief_maps'], gt_belief_maps)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()*args.batch_size*2
        num_trained_data += args.batch_size*2
    train_loss_sum /= num_trained_data
    return train_loss_sum


if __name__ == '__main__':
    # parse arguments
    args = argparse_args()
    if args is None:
        exit()
    print(args)

    main(args)