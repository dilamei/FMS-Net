import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import argparse
import logging
import random
import warnings
from datetime import datetime
from pathlib import Path
from tqdm import tqdm 

from model.PVTWithBERT import PVTwithBERT
from data import get_loader
import pytorch_iou

warnings.filterwarnings("ignore")

def clip_gradient(optimizer, grad_clip):
    """Clips gradients computed during backpropagation."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """Decay learning rate by a factor of decay_rate every decay_epoch epochs."""
    if epoch < decay_epoch:
        current_lr = init_lr
    else:
        decay_times = epoch // decay_epoch
        decay = decay_rate ** decay_times
        current_lr = init_lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    print('Epoch: {}, decay_epoch: {}, Current_LR: {}'.format(epoch, decay_epoch, current_lr))

class Trainer:
    DATASET_PATHS = {
        'ORSSD': {
            'image_root': '/data/dlm/dataset/ORSSD/train-images/',
            'gt_root': '/data/dlm/dataset/ORSSD/train-labels/',
            'text_file': '/data/dlm/dataset/ORSSD/train_descriptions.json'
        },
        'EORSSD': {
            'image_root': '/data/dlm/dataset/EORSSD/train-images/',
            'gt_root': '/data/dlm/dataset/EORSSD/train-labels/',
            'text_file': '/data/dlm/dataset/EORSSD/train_descriptions.json'
        },
        'ORSI-4199': {
            'image_root': '/data/dlm/dataset/ors-4199/trainset/images/',
            'gt_root': '/data/dlm/dataset/ors-4199/trainset/gt/',
            'text_file': '/data/dlm/dataset/ors-4199/train_descriptions.json'
        }
    }

    def __init__(self, args):
        """Initialize trainer with given arguments."""
        self.args = args
        self.device = self.setup_device()
        self.logger = self.setup_logger()
        
        # Set random seed
        self.set_seed(args.seed)
    
        # Setup components
        self.setup_paths()
        self.model = self.setup_model()
        self.train_loader = self.setup_dataloader()
        self.total_step = len(self.train_loader)
    
        self.optimizer = self.setup_optimizer()
        self.setup_loss_functions()

    def setup_device(self):
        """Setup computing device."""
        if torch.cuda.is_available():
            torch.cuda.set_device(1)
            return torch.device('cuda')
        return torch.device('cpu')

    def setup_logger(self):
        """Setup logging configuration."""
        log_dir = Path(f'./results/{self.args.dataset}/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def setup_paths(self):
        """Setup data and save paths."""
        dataset_paths = self.DATASET_PATHS.get(self.args.dataset)
        if not dataset_paths:
            raise ValueError(f"Dataset {self.args.dataset} not found. "
                           f"Available datasets: {list(self.DATASET_PATHS.keys())}")
        
        self.args.image_root = dataset_paths['image_root']
        self.args.gt_root = dataset_paths['gt_root']
        self.args.text_file = dataset_paths['text_file']
        self.save_path = Path(f'./results/{self.args.dataset}')
        self.save_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def set_seed(seed):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setup_model(self):
        """Setup and initialize model."""
        model = PVTwithBERT().to(self.device)
        self.logger.info(f"Model initialized on {self.device}")
        return model

    def setup_optimizer(self):
        """Setup optimizer."""
        return torch.optim.Adam(
            self.model.parameters(), 
            self.args.lr
        )
    
    def setup_loss_functions(self):
        """Setup loss functions."""
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.criterion_iou = pytorch_iou.IOU(size_average=True)
        self.sigmoid = torch.nn.Sigmoid()

    def setup_dataloader(self):
        """Setup data loader."""
        return get_loader(
            image_root=self.args.image_root,
            gt_root=self.args.gt_root,
            text_file=self.args.text_file,
            batchsize=self.args.batchsize,
            trainsize=self.args.trainsize,
            shuffle=False,
            augment=False,
            cache_text=False
        )

    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_name = self.save_path / f'checkpoint_epoch_{epoch}.pth'
        torch.save(self.model.state_dict(), checkpoint_name)
        self.logger.info(f'Saved checkpoint: {checkpoint_name}')

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
#        self.set_seed(self.args.seed)

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epoch}', position=1, leave=False)
    
        for i, (images, gts, text) in enumerate(pbar, start=1):
            # Move data to device
            images = Variable(images).to(self.device)
            gts = Variable(gts).to(self.device)
            text = Variable(text).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            s1, s2, s3, s4 = self.model(images, text)
            
            # Calculate losses
            loss1 = self.criterion_bce(s1, gts) + self.criterion_iou(self.sigmoid(s1), gts)
            loss2 = self.criterion_bce(s2, gts) + self.criterion_iou(self.sigmoid(s2), gts)
            loss3 = self.criterion_bce(s3, gts) + self.criterion_iou(self.sigmoid(s3), gts)
            loss4 = self.criterion_bce(s4, gts) + self.criterion_iou(self.sigmoid(s4), gts)
            
            loss = loss1 + loss2 + loss3 + loss4
            
            # Backward pass
            loss.backward()
            clip_gradient(self.optimizer, self.args.clip)
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            avg_loss = total_loss / i

            pbar.set_postfix({
                'Avg Loss': f'{avg_loss:.4f}',
                'Loss1': f'{loss1.item():.4f}',
                'Loss2': f'{loss2.item():.4f}',
                'Loss3': f'{loss3.item():.4f}',
                'Loss4': f'{loss4.item():.4f}'
            })

        pbar.close()
        
        return avg_loss
    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training on {self.args.dataset} dataset...")
        self.logger.info(f"Training size: {self.args.trainsize}, Batch size: {self.args.batchsize}")
        
        best_loss = float('inf')
        
        # Create epoch progress bar
        epoch_pbar = tqdm(range(1, self.args.epoch + 1), desc="Training Progress", position=0)
        
        for epoch in epoch_pbar:
            adjust_lr(
                self.optimizer, 
                self.args.lr, 
                epoch, 
                self.args.decay_rate, 
                self.args.decay_epoch
            )
            
            avg_loss = self.train_epoch(epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch)
            
            # Update best loss info
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch
                self.logger.info(f'New best loss at epoch {epoch}: {best_loss:.4f}')
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({"Best Loss": f"{best_loss:.4f}"})
        
        # Report final results
        self.logger.info("Training Completed!")
        self.logger.info(f"Best Loss: {best_loss:.4f} at epoch {best_epoch}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PVTwithBERT model')
    
    # Training settings
    parser.add_argument('--epoch', type=int, default=100,
                      help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='learning rate')
    parser.add_argument('--batchsize', type=int, default=32,
                      help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352,
                      help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5,
                      help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, 
                      help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, 
                      help='every n epochs decay learning rate')
    
    # Dataset settings
    parser.add_argument('--dataset', type=str, default='ORSSD',
                      choices=['ORSSD', 'EORSSD', 'ORSI-4199'],
                      help='dataset to use for training')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=10,
                      help='random seed')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()