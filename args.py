import argparse


parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--save_path', type=str, default='exp/epilepsy/test')
parser.add_argument('--dataset', type=str, default='har')
parser.add_argument('--UCR_folder', type=str, default='HAR')
parser.add_argument('--data_path', type=str,
                    default='data/')
parser.add_argument('--freeze', type=bool,default=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--train_batch_size', type=int, default=50)#200
parser.add_argument('--test_batch_size', type=int, default=50)
parser.add_argument('--adjust',  type=bool, default=False)
parser.add_argument('--targetdata',  type=str, default='NATOPS')
# model args
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--attn_heads', type=int, default=4)
parser.add_argument('--eval_per_steps', type=int, default=16)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--layers', type=int, default=8)
parser.add_argument('--alpha', type=float, default=5.0)
parser.add_argument('--beta', type=float, default=1.0)

parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--vocab_size', type=int, default=192)
parser.add_argument('--wave_length', type=int, default=16)
parser.add_argument('--mask_ratio', type=float, default=0.5)
parser.add_argument('--reg_layers', type=int, default=4)

# train args
parser.add_argument('--lr', type=float, default=0.001)#0.001
parser.add_argument('--lr_decay_rate', type=float, default=1.)
parser.add_argument('--lr_decay_steps', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--num_epoch_pretrain', type=int, default=50)
parser.add_argument('--num_epoch', type=int, default=150)
parser.add_argument('--load_pretrained_model', type=int, default=1)
parser.add_argument('--bigker', default=True)


args = parser.parse_args()
