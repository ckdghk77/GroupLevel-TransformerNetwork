from __future__ import division
from __future__ import print_function

import argparse
import datetime
from modules.modules_GLTN import *
import random
import math


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed.')   ##  59 good
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.c')
parser.add_argument('--lr-decay', type=int, default=100,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=1,
                    help='LR decay factor.')
parser.add_argument('--batch-size', type=int, default=120,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--save-pixel-offset', type=bool, default=False,
                    help='check for saving pixel-offset. only works with _generate with interpolation_')
parser.add_argument('--sequence-num', type=int, default=5,
                    help='sequence num')
parser.add_argument('--graph-sequence', type=int, default=5,
                    help='graph sequence num')
parser.add_argument('--target-class', type=int, default=8,
                    help='target_class')
parser.add_argument('--exp-type', type=int, default=4,
                    help='exp-type (e.g 0 for moon , 1 for spiral ,  '
                         '2 for mnist,  3 for omniglot, 4 for fmnist, 5 for cifar'
                         '6 for custom)')
parser.add_argument('--data-num', type=int, default=5,
                    help='data number')
parser.add_argument('--is-rgb', type=bool, default=True);
parser.add_argument('--transparent', type=bool, default=False);
parser.add_argument('--out-folder', type=str, default='./out')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

now = datetime.datetime.now()
timestamp = now.isoformat()


# load model and meta-data

if args.exp_type == 1:
    load_folder = '{}/exp_toy_mnist_prop_{}_{}_move/'.format('epoch_result', args.data_num, args.seed);
elif args.exp_type == 2:
    load_folder = '{}/exp_mnist_prop_{}_{}/'.format('epoch_result',args.data_num, args.seed);
elif args.exp_type == 3:
    load_folder = '{}/exp_omniglot_prop_{}_{}/'.format('epoch_result', args.data_num, args.seed);
elif args.exp_type == 4:
    load_folder = '{}/exp_fmnist_prop_{}_{}/'.format('epoch_result',args.data_num, args.seed);
elif args.exp_type == 5:
    load_folder = '{}/exp_cifar_prop_{}_{}/'.format('epoch_result',args.data_num, args.seed);
elif args.exp_type == 6:
    load_folder = '{}/exp_custom_prop_{}_{}/'.format('epoch_result',args.data_num, args.seed);

try :
    encoder_file = os.path.join(load_folder, 'Encoder.pt')
    transformer_file = os.path.join(load_folder, 'Transformer.pt')

except :
    print("WARNING: No load_folder provided!" +
          "Testing (within this script) will throw an error.")
    exit(0)


# load dataset
if args.exp_type == 1 : # toy_mnist
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_toy_mnist_sequence(
        args.batch_size, train_size=args.data_num, val_size=10, test_size=10,
        sequence_num=args.sequence_num, target_class=args.target_class, seed=args.seed, c_mode='move');

elif args.exp_type == 2 : # mnist
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_mnist_sequence(
        args.batch_size, train_size= args.data_num, val_size=10, test_size=10,
        sequence_num=args.sequence_num, target_class=args.target_class, seed=args.seed);
elif args.exp_type == 3 : # omniglot
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_omniglot_sequence(
        args.batch_size, train_size=args.data_num, val_size=10, test_size=10,
        sequence_num=args.sequence_num, target_class=[args.target_class], seed=args.seed);

elif args.exp_type == 4: # fmnist
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_fmnist_sequence(
        args.batch_size, train_size=args.data_num, val_size=10, test_size=10,
        sequence_num=args.sequence_num, target_class=args.target_class, seed=args.seed);
elif args.exp_type == 5:  # cifar
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_cifar_sequence(
        args.batch_size, train_size=args.data_num, val_size=10, test_size=10,
        sequence_num=args.sequence_num, target_class=args.target_class, seed=args.seed);
elif args.exp_type == 6:  # custom
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_custom_sequence(
        args.batch_size, train_size=args.data_num, sequence_num=args.sequence_num, seed=args.seed,
        transparent=args.transparent, is_rgb=args.is_rgb);

off_diag = np.ones([args.sequence_num,args.sequence_num]) - np.eye(args.sequence_num)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)

rel_rec = torch.FloatTensor(rel_rec)
rel_full = torch.FloatTensor(off_diag)
rel_send = torch.FloatTensor(rel_send)

# Load Model
if args.exp_type == 1 :  # toy_mnist
    Encoder_net = Encoder_graph(-1, 36, 12, args.sequence_num, type='mnist_conv');
    Transformer_net = Transformer(-1, 36, 12, args.sequence_num, type='mnist_conv');
elif args.exp_type == 2 : # mnist model
    Encoder_net = Encoder_graph(-1, 36, 12, args.sequence_num, type='mnist_conv');
    Transformer_net = Transformer(-1, 36, 12, args.sequence_num, type='mnist_conv');
elif args.exp_type == 3 : # omniglot model
    Encoder_net = Encoder_graph(-1, 72, 24, args.sequence_num, type='omniglot_conv');
    Transformer_net = Transformer(-1, 72, 24, args.sequence_num, type='omniglot_conv');
elif args.exp_type == 4 : # fashion mnist model  use same with omniglot
    Encoder_net = Encoder_graph(-1, 72, 18, args.sequence_num, type='fmnist_conv');
    Transformer_net = Transformer(-1, 72, 18, args.sequence_num, type='fmnist_conv');
elif args.exp_type == 5 : # cifar model
    Encoder_net = Encoder_graph(-1, 256, 32, args.sequence_num, type='cifar_conv');
    Transformer_net = Transformer(-1, 256, 32, args.sequence_num, type='cifar_conv');
elif args.exp_type == 6 and args.is_rgb == False: # custom_gray model
    Encoder_net = Encoder_graph(-1, 512, 64,args.sequence_num, type='custom_gray_conv');
    Transformer_net = Transformer(-1, 512, 64, args.sequence_num, type='custom_gray_conv');
elif args.exp_type == 6 and args.is_rgb == True : # custom_color model
    Encoder_net = Encoder_graph(-1, 512, 64,args.sequence_num, type='custom_rgb_conv');
    Transformer_net = Transformer(-1, 512, 64, args.sequence_num, type='custom_rgb_conv');


Encoder_net.load_state_dict(torch.load(encoder_file));
Transformer_net.load_state_dict(torch.load(transformer_file));


if args.cuda:
    Encoder_net = Encoder_net.cuda()
    Transformer_net = Transformer_net.cuda()
    rel_rec = rel_rec.cuda();
    rel_full = rel_full.cuda();
    rel_send = rel_send.cuda();

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)
rel_full = Variable(rel_full)


origin_train_list = list();
target_train_list = list();

final_transform_list = list();

pixel_offset_list = list();
value_offset_list = list();

new_generated_list = list();

# train_data_list
train_src_list = list();

for i in range(train_data_list.shape[0]) :

    total_target_list = list();
    for j in range(train_data_list.shape[0]) :
        if i==j :
            continue;

        total_target_list.append(train_data_list[j]);

    src_target_list = list();
    for j in range(math.ceil(len(total_target_list)/(args.sequence_num-1))) :
        each_target_list = list();
        each_target_list.append(train_data_list[i]);

        if (j+1)*(args.sequence_num-1) > len(total_target_list) :
            each_target_list.extend(total_target_list[j*(args.sequence_num-1):]);
            each_target_list.extend(total_target_list[:((j+1)*(args.sequence_num-1) - len(total_target_list))]);
        else :
            each_target_list.extend(total_target_list[j*(args.sequence_num-1):(j+1)*(args.sequence_num-1)]);

        src_target_list.append(each_target_list);
    train_src_list.extend(src_target_list);


if len(train_src_list) > 100 :
    random.shuffle(train_src_list);

generate_src_data = torch.FloatTensor(np.stack(train_src_list))[:100]; # if too many

with torch.no_grad():

    data = generate_src_data.contiguous()
    Encoder_net.eval()
    Transformer_net.eval()

    if args.cuda:
        data = data.cuda()

    data = Variable(data);

    transform_parameter = Encoder_net(data, rel_rec, rel_send, rel_full, sequence_num=args.sequence_num);

    preds, identical_preds, value_offset, pixel_offset = Transformer_net(data[:, 0], rel_rec, rel_send,
                                                                         transform_parameter,
                                                                         iter_num=args.graph_sequence,
                                                                         sequence_num=args.sequence_num);

    #new_data, new_data_intermet = Decoder_net.generate_with_random(data[:,0], rel_rec, rel_send, edges, iter_num = args.graph_sequence, sequence_num = args.sequence_num,
    #                                                       random_normal=[0.0, 0.05]);

    new_data, value_offset, pixel_offset = Transformer_net.generate_with_interpolation(data[:,0], rel_rec, rel_send, transform_parameter, iter_num = args.graph_sequence, sequence_num = args.sequence_num,
                                                          interpol_range=[0.1,1.0], gen_num= 23);


    origin_train_list.append(data.data.cpu().numpy())
    target_train_list.append(data.data.cpu().numpy())
    new_generated_list.append(new_data.data.cpu().numpy());
    if args.save_pixel_offset :
        pixel_offset_list.append(pixel_offset.data.cpu().numpy());
        value_offset_list.append(value_offset.data.cpu().numpy());

    final_transform_list.append(preds.data.cpu().numpy());


np.save(args.out_folder + '/train_x', np.vstack(origin_train_list));
np.save(args.out_folder + '/train_target' , np.vstack(target_train_list));
np.save(args.out_folder + '/total_train_x' , np.stack(train_data_list));

np.save(args.out_folder + '/train_transform', np.stack(final_transform_list));

np.save(args.out_folder + '/generated' , np.vstack(new_generated_list));
if args.save_pixel_offset :
    np.save(args.out_folder + '/pixel_offset' , np.vstack(pixel_offset_list));
    np.save(args.out_folder + '/value_offset' , np.vstack(value_offset_list));

print("Inference Finished!")




