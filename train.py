import os
import sys
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch.nn as nn

from Data_Preprocessing import PreprocessBatch
import time

class Trainer(object):
    '''
    The Trainer class is used to train, test, validate the model as well as save its performance in appropriate locations. 
    '''
    def __init__(self, path, model, train_img_paths, test_img_paths, batch_size, num_epochs,
                 criterion, optimizer, lr_scheduler):
        self.path = path
        self.model = model
        self.set_device()

        self.test_img_paths = test_img_paths
        self.train_img_paths = train_img_paths
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.initialize_folder()
        self.initialize_logging()

        self.train_loss_epoch = []
        self.train_acc = []
        
        self.val_loss_epoch = []
        self.val_acc = []

        self.lr_values = []

    def initialize_folder(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def initialize_logging(self):
        logging.basicConfig(filename=self.path + '/train_log.log',
                            format='[%(asctime)s] %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            filemode='w')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)


    def set_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device == torch.device('cuda'):
            gpu_id = self.find_best_gpu()
            if gpu_id:
                torch.cuda.set_device(gpu_id)
        self.device = device
        print('Device selected: ', self.device, torch.cuda.current_device())
        self.model = self.model.to(self.device)
        self.model.device = self.device

    def find_best_gpu(self):
        if 'linux' in sys.platform and torch.cuda.device_count() > 1:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            gpu_id = np.argmax(memory_available).item()
            print('best gpu, %d, %f' % (gpu_id, memory_available[gpu_id]))

            return gpu_id

    def get_accuracy(self, preds, gt):
        correct = 0
        total = 0
        preds = torch.max((preds),1)[1]
        for i in range(len(preds)):
            if preds[i] == gt[i]:
                correct += 1
            total +=1
        return correct

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            epoch_loss_val = []
            acc_val = []
            for batch_num in range(0,len(self.test_img_paths),self.batch_size):
                X, Y = PreprocessBatch(self.test_img_paths[batch_num:batch_num+self.batch_size])
                X, Y = X.to(self.device).float(), Y.to(self.device).long()
                
                pred = self.model(X)
                
                loss = self.criterion(pred,Y)
                
                epoch_loss_val.append(loss.item())
                acc_val.append(self.get_accuracy(pred,Y))
                
            return sum(epoch_loss_val)/len(epoch_loss_val), sum(acc_val)/(len(self.test_img_paths))

    def save_val_set_outputs(self):
        weight_path = self.path + '/model_weights.pth'
        self.load_model_weights(weight_path)
        self.model.eval()
        gt_pred = []
        with torch.no_grad():
            for batch_num in range(0,len(self.test_img_paths),self.batch_size):
                X, Y = PreprocessBatch(self.test_img_paths[batch_num:batch_num+self.batch_size])
                X, Y = X.to(self.device).float(), Y.to(self.device).long()
                
                preds = self.model(X)
                gt_pred.append([Y.detach().cpu().numpy(), preds.detach().cpu().numpy()])
                
                del X, Y, preds
                torch.cuda.empty_cache()

        np.save(self.path + '/gt_pred.npy', np.array(gt_pred))

        

    def train(self):
        prev_loss = 1e10
        for epoch in range(self.num_epochs):
            tic = time.time()
            self.model.train()
            np.random.shuffle(self.train_img_paths)
            epoch_loss = []
            epoch_acc = []
            for batch_num in range(0,len(self.train_img_paths), self.batch_size):
                X, Y = PreprocessBatch(self.train_img_paths[batch_num:batch_num+self.batch_size])
                X, Y = X.to(self.device).float(), Y.to(self.device).long()     
                self.optimizer.zero_grad()

                pred = self.model(X)
                
                loss = self.criterion(pred,Y)
                loss.backward()

                self.optimizer.step()

                epoch_loss.append(loss.item())
                epoch_acc.append(self.get_accuracy(pred,Y))
            
            
            val_loss, val_acc = self.validate()

            if loss.item() < prev_loss:
                self.save_model(epoch)
                prev_loss = loss.item()
            
            self.train_loss_epoch.append(np.mean(epoch_loss))
            self.val_loss_epoch.append(val_loss)

            self.train_acc.append(sum(epoch_acc)/(len(self.train_img_paths)))
            self.val_acc.append(val_acc)

            self.lr_values.append(self.optimizer.param_groups[0]['lr'])
            
            self.save_files()
            self.lr_scheduler.step()

            epoch_time = time.time() - tic
            self.log_epoch_stat(epoch, epoch_time)

        state = {'epoch': epoch,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.path + '/final_weights.pth')
        self.save_val_set_outputs()



    def log_epoch_stat(self, epoch, epoch_time):
        self.logger.info(f'\n Epoch number: {epoch}\
        \n Train loss: {self.train_loss_epoch[-1]}\
        \n Validation loss: {self.val_loss_epoch[-1]}\
        \n Train Accuracy : {self.train_acc[-1]}\
        \n Validation Accuracy : {self.val_acc[-1]}\
        \n Learning Rate: {self.lr_values[-1]}\
        \n Time: {epoch_time} \n')



    def set_device_manual(self, gpu_id):
        if self.device == torch.device('cuda'):
            torch.cuda.set_device(gpu_id)
            self.model = self.model.to(self.device)
            print('GPU manually changed to ' + str(gpu_id))



    def load_model_weights(self, address):
        print('Loading model weights')
        states = torch.load(address)
        model_weights = states['model']
        try:
            self.model.load_state_dict(model_weights)
        except:
            modified_dict = dict()
            for key, val in model_weights.items():
                if 'module' in key:
                    modified_dict[key.strip('module.')] = val
                else:
                    modified_dict[key] = val
            self.model.load_state_dict(modified_dict)

    def load_optimizer_states(self, address, lr=0.000005):
        print('Loading optimizer states')
        states = torch.load(address)
        opt_state = states['optimizer']
        self.optimizer.load_state_dict(opt_state)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        self.optimizer.param_groups[0]['lr'] = lr


    def save_model(self, epoch):
        state = {'epoch': epoch,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, self.path + '/model_weights.pth')
        self.logger.info(f'\n Saving weights, epoch number: {epoch} \n')

    def save_files(self):
        np.save(self.path + '/train_loss_epoch.npy', np.array(self.train_loss_epoch))
        np.save(self.path + '/train_accuracy.npy', np.array(self.train_acc))
        np.save(self.path + '/val_loss_epoch.npy', np.array(self.val_loss_epoch))
        np.save(self.path + '/val_accuracy.npy', np.array(self.val_acc))
        np.save(self.path + '/lr_values.npy', np.array(self.lr_values))
        self.save_graphs()

    def save_graphs(self):
        plt.figure(figsize=(9, 6))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(np.log10(self.train_loss_epoch), label='Train loss')
        plt.plot(np.log10(self.val_loss_epoch), label='Validation loss')
        plt.title('Loss vs epoch', fontsize=18)
        plt.xlabel('Number of epochs', fontsize=18)
        plt.ylabel('Mean Square Error', fontsize=18)
        plt.legend()
        plt.savefig(self.path + '/loss.png', bbox_inches='tight', dpi=600)
        plt.close()

        plt.figure(figsize=(9, 6))
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.plot(self.train_acc, label='Train Accuracy')
        plt.plot(self.val_acc, label='Validation Accuracy')
        plt.title('Accuracy vs epoch', fontsize=18)
        plt.xlabel('Number of epochs', fontsize=18)
        plt.ylabel('Accuracy', fontsize=18)
        plt.legend()
        plt.savefig(self.path + '/accuracy.png', bbox_inches='tight', dpi=600)
        plt.close()


    
    def train_model(self, device_ids):
        if len(device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            self.model = self.model.to(self.device)
            print('Training on multiple GPUs using DataParallel module...')
            self.train()
        else:
            self.train()


