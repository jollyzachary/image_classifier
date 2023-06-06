from imports import *

def save_checkpoint(model, classifier, criterion, train_data, optimizer, filename='checkpoint.pth'):
    # Save the checkpoint
    checkpoint = {
        'input_size': 25088,
        'output_size': 102,
        'model': models.vgg16(pretrained=True),
        'classifier': classifier,
        'criterion': criterion,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }

    torch.save(checkpoint, filename)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    criterion = checkpoint['criterion']
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model, criterion, optimizer
