import argparse, json, os
from NeuralNet import NeuralNet
import matplotlib.pyplot as plt
import seaborn as sns
from image_utils import process_image, imshow, predict


parser = argparse.ArgumentParser(description='Training a neural network')
parser.add_argument('image', type=str, help='Input image for prediction')
parser.add_argument('checkpoint', help='Provide path to model checkpoint')
parser.add_argument(
    '--top_k',
    default=3,
    type=int,
    help='Return top k most likely classes (default: 3)'
)
parser.add_argument(
    '--category_names',
    default='',
    help='Use a mapping of categories to real names (default: .)'
)
parser.add_argument(
    '-gpu',
    action="store_true",
    default=False,
    help='Use GPU for inference')

args = parser.parse_args()

net = NeuralNet(data_dir='', predict=True)
model = net.load_model(checkpoint=args.checkpoint) 


if os.path.isfile(args.category_names):
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    cat_to_name = None

# Make prediction
probs, labs, names = predict(args.image, model, top_num=args.top_k,
                               gpu=args.gpu, cat_to_name=cat_to_name)

for p, l, n in zip(probs, labs, names):
    print(f'Probability: {p} - Class: {l} - Name: {n}')
