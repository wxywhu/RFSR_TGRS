#!/usr/bin/env python
# coding: utf-8

# In[ ]:


""" Main function for this repo. """
import argparse
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import HSIlmdbDataset
from dataset import data_prefetcher
from tensorboardX import SummaryWriter
import tqdm
import os
import torch.nn as nn
import time
from RFSR import Net
import pandas as pd
import torch.optim as optim
from torchnet import meter
import torch.backends.cudnn as cudnn
from dataset import *
import scipy.io as sio
from metrics import *
import random
from util import *

def main(args):
    print(args.gpus)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.phase is None:
      print('ERROR: specify either train or test')
      sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
      print('ERROR: cuda is not available, try running on CPU')
      sys.exit(1)
    if args.phase == "train":
      print('=============================Train=============================')  
      train(args)
    else:
      print('=============================Test==============================')
      test(args)
    pass
    
def validate(args, loader, model, criterion):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval()
    epoch_meter = meter.AverageValueMeter()
    epoch_meter.reset()
    with torch.no_grad():
      for i, (lr, gt) in enumerate(loader):
          lr, gt = lr.to(device), gt.to(device)           
          pre = model(lr)
          loss = criterion(pre, gt)
          epoch_meter.add(loss.item())
    model.train()
    return epoch_meter.value()[0]

    
def train(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Start seed: ", args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
      torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True
    
    print('======> Loading datasets')
    train_dir = os.path.join(args.dataset_dir,args.dataset,'data_train')
    train_set = HSIlmdbDataset(train_dir, args.scale, args.patch_size, args.seq_len)
    train_data_loader =  DataLoader(train_set, num_workers=4, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    val_dir = os.path.join(args.dataset_dir,args.dataset,'data_val')
    val_set = HSIlmdbDataset(val_dir , args.scale, args.patch_size, args.seq_len)
    val_data_loader =  DataLoader(val_set, num_workers=1, batch_size=1, shuffle=True, pin_memory=True)    
    
    print('======> Building model')
    model = Net(args.scale, args.seq_len, device)
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume_model):
            print("=> loading checkpoint '{}'".format(args.resume_model))
            checkpoint = torch.load(args.resume_model)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(resume_model))
            
    model.to(device).train()
    L1_loss  = torch.nn.L1Loss()
    H_loss = HLoss(args.la1,args.la2)
    print("======> Setting optimizer and logger")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    epoch_meter = meter.AverageValueMeter()    
    writer = SummaryWriter('runs/'+ args.model_type+'_'+str(time.ctime()))
    
    print('======> Start training')
    best=0  
    for e in range(start_epoch, args.epoch):
      adjust_learning_rate(args.lr, optimizer, e+1)
      epoch_meter.reset()
      print("Start epoch {}, learning rate = {}".format((e+1), optimizer.param_groups[0]["lr"]))
      st= time.time()
      for step, (lr, hr) in enumerate(train_data_loader):
          lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

          optimizer.zero_grad()  
          t1 = time.time()
          pre = model(lr)
          loss = H_loss(pre,hr)
          epoch_meter.add(loss.item())
          loss.backward()
          # torch.nn.utils.clip_grad_norm(net.parameters(), clip_para)
          optimizer.step()
          t2 = time.time()          
          if step % 10 == 0:
              print("Epoch[{}]({}/{}):  loss: {:.4f} || Timer: {:.4f} sec.".format(e+1, step, len(train_data_loader), loss.item(),(t2-t1)))
              writer.add_scalar('Loss/train_loss',loss.item(), e*len(train_data_loader)+step+1)
      et=time.time()          
      print("====Epoch[{}]:  Average Loss: {:.5f} || Total time:{:.4f} sec. ".format(e+1, epoch_meter.value()[0], (et-st)))
      print('======> Validation')
      eval_loss = validate(args, val_data_loader, model, L1_loss)
      print("====Epoch[{}]:  Validation Loss: {:.5f}".format(e+1,eval_loss))
      writer.add_scalar('Loss/avg_validation_loss', eval_loss, e+1)
      
      if e==0:
        best=eval_loss
        save_checkpoint(args, model, e+1, args.save_ckp)
      else:
        if eval_loss < best:
          best=eval_loss
          save_checkpoint(args, model, e+1, args.save_ckp)
        if (e+1)%10==0:
          save_checkpoint(args, model, e+1, args.save_ckp)
      

    writer.export_scalars_to_json("./all_scalars_json")
    writer.close()
    
def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)
    print('===> Loading testset')
    hr_path = get_data_paths(os.path.join(args.test_dir,'HR'))
    lr_path = get_data_paths(os.path.join(args.test_dir,'LR','x'+str(args.noise)))
    print('===> Start testing')
    with torch.no_grad():
        # loading model
        #model = Net(args.scale, args.seq_len, device)
        if os.path.isfile(args.test_model):             
          print("=> loading checkpoint '{}'".format(args.test_model))             
          model = torch.load(args.test_model)["model"]
        else:
            print('No such model')
          
        model.to(device).eval()

        xls_list=[]
        psnr, sam, ssim, ergas, rmse, cc= [],[],[],[],[],[]
        for index in range(len(hr_path)):
            # compute output
            name = os.path.split(hr_path[index])[-1]
            hr =  single2tensor4(uint2single(sio.loadmat(hr_path[index])['data']))
            lr = single2tensor4(uint2single(sio.loadmat(lr_path[index])['LR']))
            print(lr.shape)
            lr = lr.to(device)
            pre = chop_forward(lr, model, args.scale)
            pre, hr = tensor2single(pre.clamp(0,1)), tensor2single(hr)
            if args.save_results:
                save_mat(pre, name[0:-4])
            indices = quality_assessment(pre, hr, data_range=1., ratio=args.scale)
            print("Image:{}, psnr: {:.4f} || sam: {:.4f}".format(name, indices['MPSNR'], indices['SAM']))
            psnr.append(indices['MPSNR'])
            sam.append(indices['SAM'])
            ssim.append(indices['MSSIM'])
            ergas.append(indices['ERGAS'])
            rmse.append(indices['RMSE'])
            cc.append(indices['CrossCorrelation'])
            xls_list.append([name, indices['MPSNR'], indices['SAM'], indices['MSSIM'],indices['ERGAS'],indices['RMSE'],indices['CrossCorrelation']])
        
        print("=========Test finished==========")
        print("Average:psnr:{:.4f}||sam:{:.4f}||ssim:{:.4f}||ergas:{:.4f}||rmse:{:.4f}||cc:{:.4f}".format(np.mean(psnr),np.mean(sam),np.mean(ssim),np.mean(ergas),np.mean(rmse),np.mean(cc)))    
        xls_list.append(['Average',np.mean(psnr),np.mean(sam),np.mean(ssim),np.mean(ergas),np.mean(rmse),np.mean(cc)])
        xls_list = np.array(xls_list)

        result = pd.DataFrame(xls_list, columns=['NAME','MPSNR','SAM','MSSIM','ERGAS','RMSE','CC'])
        result.to_csv(args.model_type + args.dataset + 'x'+str(args.noise)+'.csv')    
                    
def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return img

def chop_forward(x, model, scale,shave=16):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:,:, 0:h_size, 0:w_size],
        x[:,:, 0:h_size, (w - w_size):w],
        x[:,:, (h - h_size):h, 0:w_size],
        x[:,:, (h - h_size):h, (w - w_size):w]] 
    outputlist = []
    for i in range(4):
      input_batch = inputlist[i]
      output_batch = model(input_batch)
      outputlist.append(output_batch)
    
    output = Variable(x.data.new(b, c, h*scale, w*scale))
    print(output.shape)
    output[:,:, 0:h_half*scale, 0:w_half*scale] = outputlist[0][:, :, 0:h_half*scale, 0:w_half*scale]    
    output[:,:, 0:h_half*scale, w_half*scale:w*scale] = outputlist[1][:, :, 0:h_half*scale, (w_size - w + w_half)*scale:w_size*scale]
    output[:,:, h_half*scale:h*scale, 0:w_half*scale] = outputlist[2][:, :, (h_size - h + h_half)*scale:h_size*scale, 0:w_half*scale]   
    output[:,:, h_half*scale:h*scale, w_half*scale:w*scale] = outputlist[3][:, :, (h_size - h + h_half)*scale:h_size*scale, (w_size - w + w_half)*scale:w_size*scale]
    
    return output

def save_mat(img, img_name):
    save_dir=os.path.join('./Results', args.dataset,'x'+ str(args.noise))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    data=np.uint16(np.clip(img*65535,0,65535))    
    save_fn = os.path.join(save_dir, img_name +'_'+args.model_type+'.mat')
    print(save_fn)
    sio.savemat(save_fn, {'SR': data})
    
def save_checkpoint(args, model, epoch, path):
    device = torch.device("cuda" if args.cuda else "cpu")
    model.eval().cpu()
    if not os.path.exists(path):
        os.makedirs(path)
    save_filename = '{}_{}_x{}_{}.pth'.format(args.model_type, args.dataset, args.scale, epoch)
    save_path = os.path.join(path, save_filename)
    state = {"epoch": epoch, "model": model}
    torch.save(state, save_path)
    model.to(device).train()
    print("===Sucessfully save epoch {} model to {}===".format(epoch, save_path))
     
     
def load_network(network, path, strict=True):
    if isinstance(network, nn.DataParallel):
        network = network.module
    network.load_state_dict(torch.load(path), strict=strict)          
    print("===Successfully load the pre-trained model===") 
    
def adjust_learning_rate(start_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 50 epochs"""
    lr = start_lr * (0.1 ** (epoch // 160))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  
          

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='RFSR',  help="Network") #The network architecture
    parser.add_argument('--dataset', type=str, default='CAVE') #Dataset
    
    parser.add_argument('--phase', type=str, default='train') #Phase
    parser.add_argument('--seed', type=int, default=0) #Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpus', type=str, default="2") 
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--dataset_dir', type=str, default= '/data1/wxy/HSI/') #Dataset folder
    
    # Parameters for train phase
    parser.add_argument('--epoch', type=int, default=200) #Epoch number for meta-train phase
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate") 
    parser.add_argument('--batch_size', type=int, default=16, help="Learning rate") 
    parser.add_argument('--save_ckp', type=str, default='./Checkpoints', help="")
    parser.add_argument('--RGB_premodel', type=bool, default=True)
    
    parser.add_argument('--resume', type=bool, default=False, help="")
    parser.add_argument('--resume_model', type=str, default='', help="")
    
    #Parameters for image
    parser.add_argument('--la1', type=float, default=0.5, help="")
    parser.add_argument('--la2', type=float, default=0.1, help="")
    parser.add_argument('--sam', type=bool, default=False, help ='') #Phase
    parser.add_argument('--gra', type=bool, default=False, help ='') 
    
    #Parameters for image
    parser.add_argument('--scale', type=int, default=4, help="The scale factor") 
    parser.add_argument('--patch_size', type=int, default=64, help ='The HR patch size') #Phase
    parser.add_argument('--seq_len', type=int, default=31, help ='The LR patch size') 
    
    # Parameters for test phase
    parser.add_argument('--test_dir', type=str, default='/data1/wxy/HSI/', help="Test data set")  
    parser.add_argument('--test_model', type=str, default='', help="The meta model path") 
    parser.add_argument('--save_path', type=str, default='./Results', help="") 
    parser.add_argument('--save_results', type=bool, default=True, help='')
     
    # Set and print the parameters
    args = parser.parse_args()
 
    main(args)

