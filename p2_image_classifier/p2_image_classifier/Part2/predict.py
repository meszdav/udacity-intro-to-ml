#importing the libraries
import predict_funcs as pf
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("path", action = 'store', 
                    help='Give the directory of your image /path/to/image.jpg')
parser.add_argument("checkpoint",  help='Select the checkpoint file.')
parser.add_argument("--topk", action = 'store', default = 3 , type = int,
                    help='The number of listed probabilities. Default 3')
parser.add_argument("--category_names", action = 'store', default = "cat_to_name.json" ,
                    help='Useing a json to indentify the classes. Default "cat_to_name.json"')
parser.add_argument('--gpu', action='store', nargs='?',default = "cpu",
                     help='The prediction will use the gpu. Default the device uses the "cpu".', const = "cuda")


args = parser.parse_args()

image_path = args.path
checkpoint =  args.checkpoint
topk = args.topk
cat_names = args.category_names
device = args.gpu

if __name__ == "__main__":

    if device == "cuda" and not torch.cuda.is_available():
        
        print("You do not have gpu! Run the programm without --gpu")
        
    else:
        
        #loading the image
        image = pf.process_image(image_path)

        #Creating the model from the saved checkpoint
        model, optimizer = pf.load_checkpoint(checkpoint)

        #the topK probabilities and classes
        top_p,top_class = pf.predict(image,model,topk,device)

        #reading the category names
        cat_to_name = pf.cat_to_name(cat_names)

        #mapping the top K predictions
        flowers = [cat_to_name[str(x)] for x in top_class]

        for k,i in enumerate(flowers):
            print("The probability of the flowers is a '{}' is {:.2f}%.".format(i,top_p[k]*100))



