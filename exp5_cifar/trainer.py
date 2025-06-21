import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import json


class Trainer(object):
    def __init__(self, dataloader, model, checkpoint, save_dir, 
               num_epochs, batch_size, learning_rate, early_stopping_criteria, device):
        self.train_dataloader = dataloader['train']
        self.test_dataloader = dataloader['test']
        self.save_dir = save_dir
        self.model = model
        self.device = device

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode='min', factor=0.5, patience=1)
        self.train_state = {
            'done_training': False,
            'stop_early': False, 
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'early_stopping_criteria': early_stopping_criteria,
            'learning_rate': learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
            'model_filename': checkpoint}
    
    def update_train_state(self):

        # Verbose
        print ("[EPOCH]: {0} | [LR]: {1} | [TRAIN LOSS]: {2:.2f} | [TRAIN ACC]: {3:.1f}% | [VAL LOSS]: {4:.2f} | [VAL ACC]: {5:.1f}%".format(
          self.train_state['epoch_index'], self.train_state['learning_rate'], 
            self.train_state['train_loss'][-1], self.train_state['train_acc'][-1], 
            self.train_state['val_loss'][-1], self.train_state['val_acc'][-1]))

        # Save one model at least
        if self.train_state['epoch_index'] == 0:
            torch.save(self.model.state_dict(), self.train_state['model_filename'])
            self.train_state['stop_early'] = False

        # Save model if performance improved
        elif self.train_state['epoch_index'] >= 1:
            loss_tm1, loss_t = self.train_state['val_loss'][-2:]

            # If loss worsened
            if loss_t >= self.train_state['early_stopping_best_val']:
                # Update step
                self.train_state['early_stopping_step'] += 1

            # Loss decreased
            else:
                # Save the best model
                if loss_t < self.train_state['early_stopping_best_val']:
                    torch.save(self.model.state_dict(), self.train_state['model_filename'])

                # Reset early stopping step
                self.train_state['early_stopping_step'] = 0

            # Stop early ?
            self.train_state['stop_early'] = self.train_state['early_stopping_step'] \
              >= self.train_state['early_stopping_criteria']
        return self.train_state
  
    # def compute_accuracy(self, outputs, labels):
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += labels.size(0)
    #     correct += (predicted == labels).sum().item()
    
        
  
    def run_train_loop(self):
        for epoch_index in range(self.num_epochs):
            self.train_state['epoch_index'] = epoch_index
            # initialize batch generator, set loss and acc to 0, set train mode on
            running_loss = 0.0
            running_acc = 0.0
            self.model.train()
            total = 0
            correct = 0
            for batch_index, (images, labels) in enumerate(self.train_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                loss_t  = loss.item()

                running_loss += (loss_t - running_loss) / (batch_index + 1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            running_acc = 100 * correct / total
            self.train_state['train_loss'].append(running_loss)
            self.train_state['train_acc'].append(running_acc)                



            # initialize batch generator, set loss and acc to 0; set eval mode on

            running_loss = 0.
            running_acc = 0.
            self.model.eval()

            total = 0
            correct = 0
            for batch_index, (images, labels) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                loss_t = loss.to("cpu").item()
                
                running_loss += (loss_t - running_loss) / (batch_index + 1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            running_acc = 100 * correct / total
            self.train_state['val_loss'].append(running_loss)
            self.train_state['val_acc'].append(running_acc)

            self.train_state = self.update_train_state()
            self.scheduler.step(self.train_state['val_loss'][-1])


            if self.train_state['stop_early']:
                break
          
    def run_test_loop(self):

        running_loss = 0.0
        running_acc = 0.0
        self.model.eval()


        total = 0
        correct = 0

        for batch_index, (images, labels) in enumerate(self.test_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            loss_t = loss.to("cpu").item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        running_acc = 100 * correct / total
        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc
        # self.train_state = self.update_train_state()
        print("[TEST] | [TEST LOSS]: {0:.2f} | [TEST ACC]: {1:.1f}%".format(running_loss, running_acc))
    
    def plot_performance(self):
        # Figure size
        plt.figure(figsize=(15,5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.plot(self.train_state["train_loss"], label="train")
        plt.plot(self.train_state["val_loss"], label="val")
        plt.legend(loc='upper right')

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.plot(self.train_state["train_acc"], label="train")
        plt.plot(self.train_state["val_acc"], label="val")
        plt.legend(loc='lower right')

        # Save figure
        plt.savefig(os.path.join(self.save_dir, "performance.png"))

        # Show plots
        plt.show()
    
    def save_train_state(self):
        self.train_state["done_training"] = True
        with open(os.path.join(self.save_dir, "train_state.json"), "w") as fp:
            json.dump(self.train_state, fp)