import argparse
import data_preprocessing
import label_mapping
import model
import testing
import checkpoint
import image_processing

def main():
    parser = argparse.ArgumentParser(description='Train a model or make a prediction.')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--predict', type=str, help='Make a prediction')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint')
    # Add other command line arguments here
    args = parser.parse_args()

    if args.train:
        # Load and preprocess the image dataset
        trainloader, validloader, testloader = data_preprocessing.load_and_preprocess(args.data_directory)

        # Build and train the classifier
        model, criterion, optimizer = model.build_and_train_classifier(trainloader, validloader, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)

        # Test the network
        testing.test_network(model, criterion, testloader)

        # Save the checkpoint
        checkpoint.save_checkpoint(model, model.classifier, criterion, trainloader, optimizer, args.save_dir + '/checkpoint.pth')


    elif args.predict:
        # Load the checkpoint
        model, criterion, optimizer = checkpoint.load_checkpoint(args.checkpoint)

        # Image processing
        tensor_image, pil_image = image_processing.process_image(args.predict)
        image_processing.imshow(tensor_image)

        # Class prediction
        probs, classes, flowers = model.predict(tensor_image, model)

        # Sanity checking
        image_processing.display_image_predictions(pil_image, probs, flowers)


if __name__ == "__main__":
    main()
