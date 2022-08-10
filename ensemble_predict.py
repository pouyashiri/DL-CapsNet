import pickle
import argparse
from load_datasets import *


parser = argparse.ArgumentParser(description="Ensemble DeepCFC.")
parser.add_argument('--n_ens', default=7, required=True, type=int)
parser.add_argument('--dset', default='CIFAR10', required=True)
parser.add_argument('--drp_rate', default='0.0')
parser.add_argument('--res_folder', default='results', required=True)

args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = load(args.dset)
true_labels = np.argmax(y_test, 1)

# print(np.shape(y_test))
tot = 0*np.shape(y_test)[0]
folder = f'{args.res_folder}/{args.dset}-{args.drp_rate}'
sum_p=0
for i in range(1,args.n_ens+1):
    data = pickle.load(open(f"{folder}/{i}/predictions.p", "rb"))
    p = np.sum(np.argmax(data, 1) == true_labels) / y_test.shape[0]
    print(f'p{i} Test acc:', p)
    sum_p += p
    tot += data

print(f'Average acc: {sum_p/args.n_ens}')
print('Ensemble acc:', np.sum(np.argmax(tot, 1) == np.argmax(y_test, 1)) / y_test.shape[0])

