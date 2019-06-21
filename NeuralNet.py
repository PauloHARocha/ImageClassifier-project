import torch, os
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

class NeuralNet():
    def __init__(self, data_dir, arch='vgg19', learning_rate=0.001, hidden_units=4096, predict=False):
        self.arch = arch
        if not predict:
            self.image_datasets, self.dataloaders = self.init_data(data_dir)
            self.model, self.criterion, self.optimizer = self.init_model(hidden_units, learning_rate)
        
    def init_data(self, data_dir):
        # TODO: Define your transforms for the training and validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(45),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ])
        }

        dirs = {'train': data_dir + '/train', 
                'valid': data_dir + '/valid', }

        image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
        class_names = image_datasets['train'].classes 
        
        return image_datasets, dataloaders
    
    def init_model(self, hidden_units, learning_rate):
        self.hidden_units = hidden_units
        
        model = self.get_model(self.arch.lower())
            
        for param in model.parameters():# Freeze parameters so we don't backprop through them
            param.requires_grad = False
            
        model.classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, self.hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(self.hidden_units, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate) # Only train the classifier parameters, feature parameters are frozen
        
        return model, criterion, optimizer
    
 
    def model_train(self, epochs=10, gpu=False):
        steps = 0
        running_loss = 0
        print_every = 5
        device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        print("running in cuda . . ." if torch.cuda.is_available() and gpu else "running in cpu . . .")
        self.model.to(device)
        for epoch in range(epochs):
            for inputs, labels in self.dataloaders['train']:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                self.optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.dataloaders['valid']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = self.model.forward(inputs)
                            batch_loss = self.criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Step {steps}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(self.dataloaders['valid']):.3f}.. "
                          f"Validation accuracy: {accuracy/len(self.dataloaders['valid']):.3f}")
                    running_loss = 0
                    self.model.train()
    
    
    def save_model(self, save_dir='.'):
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        torch.save({'arch': self.arch,
                    'state_dict': self.model.state_dict(), 
                    'class_to_idx': self.image_datasets['train'].class_to_idx,
                    'hidden_units': self.hidden_units}, f'{save_dir}/classifier.pth')     
        
    def load_model(self, checkpoint='.'):
        checkpoint = torch.load(f'{checkpoint}/classifier.pth')
        
        model = self.get_model(checkpoint['arch'])
        
        model.class_to_idx = checkpoint['class_to_idx']
    
        model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
        model.load_state_dict(checkpoint['state_dict'])
        
        return model  

    def get_model(self, name):
        if name == 'vgg19':
            return models.vgg19(pretrained=True)
        elif name == 'vgg13':
            return models.vgg13(pretrained=True)
        elif name == 'densenet':
            return models.densenet121(pretrained=True)
        elif name == 'resnet34':
            return models.resnet34(pretrained=True)
        elif name == 'resnet50':
            return models.resnet50(pretrained=True)
        else:
            return models.vgg19(pretrained=True)
