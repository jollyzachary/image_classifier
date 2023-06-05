from imports import *
from image_processing import process_image
from torchvision.models import vgg16
from torchvision.models.vgg import VGG16_Weights
import label_mapping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cat_to_name = label_mapping.load_label_mapping()

def build_and_train_model(trainloader, validloader, arch, learning_rate, hidden_units, epochs, gpu):
    # Load the pre-trained model
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    #model = models.vgg16(pretrained=True)

    # Freeze the parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    # Set the classifier for the model
    model.classifier = classifier

    # Replace the original classifier with the new classifier
    model.classifier = classifier

    # Define the criterion (Negative Log Likelihood Loss)
    criterion = nn.NLLLoss()

    # Define the optimizer (Adam optimizer)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Move the model to the device available (either cpu or cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device);

    # Train the classifier
    epochs = 5
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass, then backward pass, then update weights
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Validation step
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model, criterion, optimizer

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image).type(torch.FloatTensor)
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