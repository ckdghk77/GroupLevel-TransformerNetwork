from __future__ import division
from __future__ import print_function

import time
import argparse
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.autograd._functions


from modules.modules_GLTN import *

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
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--save-folder', type=str, default=True,
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--sequence-num', type=int, default=5,
                    help='sequence num')
parser.add_argument('--graph-sequence', type=int, default=5,
                    help='graph sequence num')
parser.add_argument('--target-class', type=int, default=8,
                    help='target_class')
parser.add_argument('--exp-type', type=int, default=4,
                    help='exp-type (e.g 0 for moon , 1 for spiral ,  '
                         '2 for mnist,  3 for omniglot, 4 for fmnist, 5 for cifar'
                         '6 for custom, 7 for custom_heavy)')
parser.add_argument('--data-num', type=int, default=5,
                    help='data number')
parser.add_argument('--is-rgb', type=bool, default=True);
parser.add_argument('--transparent', type=bool, default=False);

### gray or RGB



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    now = datetime.datetime.now()
    timestamp = now.isoformat()

    if args.exp_type == 1 : # toy_mnist
        save_epoch_folder = '{}/exp_toy_mnist_prop_{}_{}/'.format('epoch_result', args.data_num,
                                                                 args.seed);
    elif args.exp_type == 2 : # mnist dataset
        save_epoch_folder = '{}/exp_mnist_prop_{}_{}/'.format('epoch_result', args.data_num, args.seed);
    elif args.exp_type == 3 : # omniglot dataset
        save_epoch_folder = '{}/exp_omniglot_prop_{}_{}/'.format('epoch_result', args.data_num, args.seed);

    elif args.exp_type == 4:  # fashion mnist dataset
        save_epoch_folder = '{}/exp_fmnist_prop_{}_{}/'.format('epoch_result', args.data_num,  args.seed);

    elif args.exp_type == 5:  # cifar dataset
        save_epoch_folder = '{}/exp_cifar_prop_{}_{}/'.format('epoch_result', args.data_num, args.seed);

    elif args.exp_type == 6 : # custom_dataset   ==> all image scale to 64 x 64.
        save_epoch_folder = '{}/exp_custom_prop_{}_{}/'.format('epoch_result', args.data_num, args.seed);
    elif args.exp_type == 6 : # custom_heavy_dataset   ==> all image scale to 64 x 64.
        save_epoch_folder = '{}/exp_custom_heavy_prop_{}_{}/'.format('epoch_result', args.data_num, args.seed);

    try :
        os.mkdir(save_epoch_folder)
    except :
        pass

    encoder_file = os.path.join(save_epoch_folder, 'Encoder.pt')
    transformer_file = os.path.join(save_epoch_folder, 'Transformer.pt')

else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")


# Load Dataset

if args.exp_type == 1:  # toy mnist
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_toy_mnist_sequence(
        args.batch_size, train_size=args.data_num, val_size=10, test_size=10,
        sequence_num=args.sequence_num, target_class=args.target_class, seed=args.seed, c_mode='move');

elif args.exp_type == 2 :  # mnist dataset
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_mnist_sequence(
        args.batch_size, train_size= args.data_num, val_size=10, test_size=10,
        sequence_num=args.sequence_num, target_class=args.target_class, seed=args.seed);

elif args.exp_type == 3:  # omniglot dataset
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

elif args.exp_type == 6 : # custom
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_custom_sequence(
        args.batch_size, train_size=args.data_num, sequence_num=args.sequence_num, seed=args.seed, transparent= args.transparent, is_rgb=args.is_rgb);

elif args.exp_type == 7 : # custom_heavy : use same dataloader with custom dataset
    train_loader, valid_loader, train_data_list, train_label_list, val_data_list, val_label_list = load_data_custom_sequence(
        args.batch_size, train_size=args.data_num, sequence_num=args.sequence_num, seed=args.seed, transparent= args.transparent, is_rgb=args.is_rgb);


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
elif args.exp_type == 7 and args.is_rgb == True:  # custom_heavy_color model
    Encoder_net = Encoder_graph(-1, 512, 64, args.sequence_num, type='custom_heavy_rgb_conv');
    Transformer_net = Transformer(-1, 512, 64, args.sequence_num, type='custom_heavy_rgb_conv');

encoder_optimizer = optim.Adam(list(Encoder_net.parameters()),
                       lr=args.lr)
scheduler_encoder = lr_scheduler.StepLR(encoder_optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

transformer_optimizer = optim.Adam(list(Transformer_net.parameters()),
                       lr=args.lr)
scheduler_transformer = lr_scheduler.StepLR(transformer_optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)


if args.cuda:
    Encoder_net = Encoder_net.cuda()
    Transformer_net = Transformer_net.cuda()
    rel_rec = rel_rec.cuda();
    rel_full = rel_full.cuda();
    rel_send = rel_send.cuda();

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)
rel_full = Variable(rel_full)


def train(train_data_set) :

    recon_train = []
    kl_train = []

    Encoder_net.train()
    Transformer_net.train()

    scheduler_encoder.step()
    scheduler_transformer.step()

    for batch_idx, (data, label) in enumerate(train_data_set):

        data = data.contiguous()

        if args.cuda:
            data, label = data.cuda(), label.cuda()

        data, label = Variable(data), Variable(label)

        encoder_optimizer.zero_grad()
        transformer_optimizer.zero_grad()

        transform_parameter = Encoder_net(data, rel_rec, rel_send, rel_full, sequence_num = args.sequence_num);

        preds, identical_preds, value_offset, pixel_offset = Transformer_net(data[:,0], rel_rec, rel_send, transform_parameter, iter_num = args.graph_sequence, sequence_num = args.sequence_num);


        reconstruction_loss = nll_gaussian(preds, data, weighting=1.0, variance=0.1);
        init_pred_loss = nll_gaussian(identical_preds, data[:,0,:], weighting=1.0, variance=0.1);


        total_loss = reconstruction_loss + init_pred_loss

        total_loss.backward();

        encoder_optimizer.step();
        transformer_optimizer.step();

        recon_train.append(reconstruction_loss.detach().cpu().numpy());
        kl_train.append(init_pred_loss.detach().cpu().numpy());

    return  np.mean(recon_train), np.mean(kl_train)

def val(epoch, best_val_loss, first_visit, t,  recon_train, kl_train, val_data_set) :


    recon_val = []
    kl_val = []

    Encoder_net.eval()
    Transformer_net.eval()

    with torch.no_grad() :
        for batch_idx, (data, label) in enumerate(val_data_set):
            data = data.contiguous()

            if args.cuda:
                data, label= data.cuda(), label.cuda()

            transform_parameter = Encoder_net(data, rel_rec, rel_send, rel_full, sequence_num=args.sequence_num);

            preds, identical_preds, value_offset, pixel_offset = Transformer_net(data[:,0], rel_rec, rel_send, transform_parameter, iter_num = args.graph_sequence, sequence_num = args.sequence_num);

            reconstruction_loss = nll_gaussian(preds, data, weighting=1.0, variance=0.1);

            init_pred_loss = nll_gaussian(identical_preds, data[:,0,:], weighting=1.0, variance=0.1);


            total_loss = reconstruction_loss  + init_pred_loss;

            recon_val.append(reconstruction_loss.detach().data.cpu().numpy());
            kl_val.append(init_pred_loss.data.detach().cpu().numpy());

            if first_visit == False and epoch == 99:  # basic training finish
                first_visit = True;


    print('Epoch: {:04d}'.format(epoch),
          'recon_train: {:.10f}'.format(np.mean(recon_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'recon_val: {:.10f}'.format(np.mean(recon_val)),
          'kl_val: {:.10f}'.format(np.mean(kl_val)),
          'time: {:.4f}s'.format(time.time() - t))

    return np.mean(recon_val), np.mean(kl_val)

# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0

recon_train_list = list();
kl_train_list = list();
recon_val_list = list();
kl_val_list = list();


for epoch in range(args.epochs):
    train_data_set = train_loader;
    val_data_set = valid_loader;

    first_visit = False;
    t = time.time()
    recon_train, kl_train = train(train_data_set);
    recon_val, kl_val = val(epoch, best_val_loss, first_visit,t, recon_train, kl_train, val_data_set);

    recon_train_list.append(recon_train);
    kl_train_list.append(kl_train);
    recon_val_list.append(recon_val);
    kl_val_list.append(kl_val);


if args.save_folder:
    torch.save(Encoder_net.state_dict(), encoder_file)
    torch.save(Transformer_net.state_dict(), transformer_file)


print("Optimization Finished!")




