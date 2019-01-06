import argparse

parser = argparse.ArgumentParser(description='reid')

parser.add_argument('--data_path',
                    default="path/to/Market-1501-v15.09.15",
                    help='path of Market-1501-v15.09.15')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate'],
                    help='train or evaluate ')

parser.add_argument('--backbone',
                    default='resnet50',
                    choices=['resnet50', 'resnet101'],
                    help='load weights ')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--weight',
                    default='weights/model_400.pt',
                    help='load weights ')

parser.add_argument('--epoch',
                    type=int,
                    default=400,
                    help='number of epoch to train')

parser.add_argument('--lr',
                    type=float,
                    default=2e-4,
                    help='learning_rate')

parser.add_argument('--lr_scheduler',
                    type=int,
                    nargs='+',
                    default=[320, 380],
                    help='MultiStepLR')

parser.add_argument("--batchid",
                    type=int,
                    default=4,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    type=int,
                    default=4,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    type=int,
                    default=8,
                    help='the batch size for test')

parser.add_argument("--gpuid",
                    default='0,1',
                    help='gpu id')

parser.add_argument("--gamma",
                    type=float,
                    default=0.1,
                    help="lr_scheduler gamma")

parser.add_argument("--cls_num",
                    type=int,
                    default=7405,
                    help="train set id number")

opt = parser.parse_args()