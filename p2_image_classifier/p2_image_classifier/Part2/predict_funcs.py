#importing the files
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image
import numpy as np
import json
import model_funcs as mf


def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    
    
    # TODO: Process a PIL image for use in a PyTorch model
    #opening the image
    img = Image.open(image)
    
    #resizing the image
    img = img.resize((255, 255))
    #cropping
    img = img.crop((0,0,224,224))
    
    #creating a numpy array
    np_image = np.array(img)
    
    #normalize the values between 0,1
    np_image = np_image/255
    np_image = (np_image-mean)/std
    np_image = np_image.transpose(2,0,1)
    
    return np_image


def load_checkpoint(file):
    
    #create the checkpoint
    checkpoint = torch.load(file,map_location = "cpu")

    #define the pretrained model
    model = mf.pretrained(checkpoint["transfer"])
    
    #freezing the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    #defining the classifier
    model.classifier = mf.Classifier(checkpoint["input_size"],
                                  checkpoint["output_size"],
                                  checkpoint["hidden_layers"])
    
    #defining the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = checkpoint["lr"])
    
    #loading the state of the model
    model.load_state_dict(checkpoint['state_dict'])
    
    #loading the state of the optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    return model, optimizer


def predict(image, model, topk, device):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    device = torch.device(device)
        
    model.to(device)
    
    
    with torch.no_grad():
        
        #model to evaluation mode
        model.eval()    
    
        
        #convert numpy array to tensor
        image = torch.from_numpy(image)

        #cast to float [1]
        image = image.float()

        #there is no batch here, therefor a new dimension is necessary[2]
        image = image.unsqueeze_(0)
        image = image.to(device)
        #forward pass
        log_ps = model(image)

        #get probability
        ps = torch.exp(log_ps)

        #top porbabilities and top classes 
        top_p, top_class = ps.topk(topk,dim=1)
        
        top_p, top_class = top_p.cpu(), top_class.cpu()

    return top_p.view(topk).numpy(),top_class.view(topk).numpy()

def cat_to_name(cat_names):

    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name