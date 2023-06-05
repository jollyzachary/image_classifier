import argparse
from data_preprocessing import load_and_preprocess
from model import build_and_train_model
from testing import test_network
from checkpoint import save_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train.py")
    parser.add_argument('data_directory', type=str, help='Data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()

    trainloader, validloader, testloader = load_and_preprocess(args.data_directory)
    model, criterion, optimizer = build_and_train_model(trainloader, validloader, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
    test_loss, test_accuracy = test_network(model, criterion, testloader)
    save_checkpoint(model, model.classifier, criterion, trainloader.dataset, optimizer, args.save_dir + '/checkpoint.pth')
    print(f'Saving checkpoint to: {args.save_dir}/checkpoint.pth')
