import argparse
from NeuralNet import NeuralNet

parser = argparse.ArgumentParser(description='Training a neural network')
parser.add_argument('data_dir', type=str, help='Input directory of training data')
parser.add_argument(
    '--save_dir',
    default='.',
    help='Provide a path to save the model (default: .)'
)
parser.add_argument(
    '--arch',
    default='vgg19',
    help='Provide the architecture for the model (default: vgg16)'
)
parser.add_argument(
    '--learning_rate',
    default=0.001,
    help='Provide the learning rate for the model (default: 0.001)'
)
parser.add_argument(
    '--hidden_units',
    default=4096,
    help='Provide the hidden units for the model (default: 4096)'
)
parser.add_argument(
    '--epochs',
    default=10,
    type=int,
    help='Provide the number of epochs for the training (default: 10)'
)
parser.add_argument(
    '-gpu',
    action="store_true",
    default=False,
    help='Activate GPU for training')

args = parser.parse_args()

net = NeuralNet(data_dir=args.data_dir, arch=args.arch,
                learning_rate=args.learning_rate, hidden_units=args.hidden_units)

net.model_train(epochs=args.epochs, gpu=args.gpu)
net.save_model(save_dir=args.save_dir)
