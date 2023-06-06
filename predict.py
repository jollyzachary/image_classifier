import torch
import argparse
from checkpoint import load_checkpoint
from image_processing import process_image
import label_mapping

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        image, _ = process_image(image_path)  # process_image returns a tensor and a PIL image
        image = image.unsqueeze(0)
        image = image.to(device)
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p = top_p.cpu().numpy().tolist()[0]
        top_class = top_class.cpu().numpy().tolist()[0]
        idx_to_class = {value: key for key, value in model.class_to_idx.items()}
        top_class = [idx_to_class[i] for i in top_class]
        top_flowers = [cat_to_name[i] for i in top_class]
        
    return top_p, top_class, top_flowers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make a prediction with a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction')
    parser.add_argument('--category_names', type=str, help='Path to JSON file containing category names')
    args = parser.parse_args()
    
    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    # Load the checkpoint
    model, criterion, optimizer = load_checkpoint(args.checkpoint)
    model = model.to(device)

    # Load category names
    if args.category_names:
        cat_to_name = label_mapping.load_label_mapping(args.category_names)
    else:
        cat_to_name = label_mapping.load_label_mapping()

    # Class prediction
    probs, classes, flowers = predict(args.image_path, model)

    # Print the results
    print("Probabilities:", probs)
    print("Classes:", classes)
    print("Flowers:", flowers)
