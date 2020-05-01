import model_funcs as mf
import argparse
import torch


parser = argparse.ArgumentParser()

parser.add_argument("dir", action = 'store', 
                    help='Pleas give the folder with your images.')
parser.add_argument("--save_directory", action = 'store', default = "model" ,
                    help='Give the directory where you want to save your model')
parser.add_argument('--arch', action='store', default = "vgg13",
                    help='Choose a transfer model: "vgg13", "vgg16". The default model is "vgg13"')
parser.add_argument('--learning_rate', action='store', default = 0.003, type = float,
                    help='Choose the learning rate to your network. Default lr = 0.003')
parser.add_argument('--hidden_units', action='store', default = 512,type = int,
                     help='Choose the parameters for the hidden units. Default 512')
parser.add_argument('--epochs', action='store', default = 1, type = int,
                     help='The number of epochs. Default = 1')
parser.add_argument('--gpu', action='store', nargs='?',default = "cpu",
                     help='The training will use the gpu. Default the device uses the "cpu".', const = "cuda")


args = parser.parse_args()

data_directory = args.dir
save_dir = args.save_directory
epochs = args.epochs
hidden_units = args.hidden_units
learning_rate = args.learning_rate
transfer = args.arch
model = mf.pretrained(transfer)
device = args.gpu

if __name__ == '__main__':
    
    #creating the classifier object
    if device == "cuda" and not torch.cuda.is_available():
        
        print("You do not have gpu! Run the programm without --gpu")
    

    else:
        
        classifier = mf.Classifier(25088,102,hidden_units)
        
        #creating the DataLoaderobject to do data transormation and datasets and dataloaders
        dataloader = mf.DataLoader(data_directory)
        
        training_transforms, data_transforms = dataloader.data_transform()
        train_datasets, valid_datasets = dataloader.datasets(training_transforms,data_transforms)
        trainloader, validloader = dataloader.dataloaders(train_datasets, valid_datasets)

        #training the model
        model, optimizer = mf.train(trainloader,validloader,model, classifier, device, epochs,learning_rate)

        #Saveing the model                    
        mf.save_model(model,optimizer, hidden_units, epochs, learning_rate, train_datasets,save_dir,transfer)
