#importing the files
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from workspace_utils import active_session



class DataLoader():
    
    def __init__(self,data_dir):
        
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        
    def data_transform(self):
        training_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.RandomCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        #transforms for the validation
        data_transforms = transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),

        ])
        
        return training_transforms, data_transforms
    
    def datasets(self,training_transforms, data_transforms):
        
        train_datasets = datasets.ImageFolder(self.train_dir, transform=training_transforms) 
        valid_datasets = datasets.ImageFolder(self.valid_dir, transform=data_transforms)
        
        return train_datasets, valid_datasets
        
        
    
    def dataloaders(self,train_datasets,valid_datasets):
        
        trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
        
        return trainloader, validloader
        
      

#Creating the Classifier
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size,hidden_layers)
        self.fc2 = nn.Linear(hidden_layers,output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self,x):
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.log_softmax(self.fc2(x),dim=1)
        
        return x
    
def pretrained(pre_model="vgg13"):
    
    """It gives back the choosen pre trained network.
    The following networks are available:
    "vgg13","vgg16"
    """
    
    if pre_model == "vgg13":
        return models.vgg13(pretrained=True)
    
    
    elif pre_model == "vgg16":
        return models.vgg16(pretrained=True)
    
def train(trainloader,validloader, model,classifier, device,epochs = 1,learning_rate = 0.002):
    
    """ Gives back the trained model and the optimizer.
    model: output of the pretrained() function 
    classifier: the Classifier class with the parameters from the user
    epochs: default = 1
    learning_rate: default = 0.002
    
    """
    with active_session():
    
        #epochs = int(epochs)

        #if the device not defined, than the device will be the CPU
        device = torch.device(device)


        #define the classifier
        model.classifier = classifier

        #defining the criterion --> output of the network is log softmax
        criterion = nn.NLLLoss()

        #creating the optimizer
        optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate)

        #initial step
        step = 0

        #using cuda if available
        model.to(device)

        #model need to be in train mode
        model.train()

        for e in range(epochs):

            #keep tracking train losses
            running_loss = 0

            for inputs, labels in trainloader:

                #keep tracking steps
                step +=1

                #using cuda if available
                inputs, labels = inputs.to(device), labels.to(device)

                #zero grad !
                optimizer.zero_grad()

                #forward pass
                log_ps = model.forward(inputs)

                #calculate loss
                loss = criterion(log_ps,labels)

                #backward pass
                loss.backward()

                #update weights
                optimizer.step()

                #increment losses
                running_loss += loss.item()

                # in each 20 steps validate the accuracy
                if step % 20== 0:        

                    #switch to evaluating mode
                    model.eval()

                    with torch.no_grad():

                        valid_loss = 0
                        accuracy=0

                        for inputs, labels in validloader:


                            #using cuda if available
                            inputs, labels = inputs.to(device), labels.to(device)

                            #forward pass
                            log_ps = model.forward(inputs)

                            #calculate testing loss
                            valid_loss += criterion(log_ps,labels).item()

                            #class probability
                            ps = torch.exp(log_ps)

                            #top classes
                            top_p,top_class = ps.topk(1,dim=1)

                            #1 if the prediction is correct
                            equals = top_class == labels.view(*top_class.shape)

                            #keep track of accuracy
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                        #print out the accuracy
                        print("Epochs: {}/{}; ".format(e+1,epochs))
                        print("Accuracy: {:.2f}%; ".format(accuracy/len(validloader)*100),
                              "Training loss: {:.3f}; ".format(running_loss/20),
                              "Validation loss: {:.3f}; ".format(valid_loss/len(validloader)))

                        #switch back to training mode
                        model.train()

                        running_loss = 0


    return model, optimizer    


def save_model(model,optimizer, hidden_units, epochs, learning_rate, train_datasets,save_dir,transfer):
    
    model.class_to_idx = train_datasets.class_to_idx
    
    checkpoint = {"input_size" : 25088,
                 "output_size" : 102,
                 "epochs" : epochs,
                  "lr" : learning_rate,
                  "hidden_layers" : hidden_units,
                 'state_dict': model.state_dict(),
                 "optimizer_state" : optimizer.state_dict(),
                 "classes" : model.class_to_idx,
                 "transfer" : transfer}


    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    
    
    
    
    
    
    
    