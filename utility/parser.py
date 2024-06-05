import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run ")
    parser.add_argument('--weights_path',
                        nargs='?',
                        default='model/',
                        help='Store model path.')
    parser.add_argument('--data_path',
                        nargs='?',
                        default='data/',
                        help='Input data path.')

    parser.add_argument('--dataset',
                        nargs='?',
                        default='origin',
                        help='Choose a dataset from {origin, remain, noised}')
    parser.add_argument('--dataset1',
                        nargs='?',
                        default='origin',
                        help='Choose a dataset from {origin, remain, forget}')

    parser.add_argument('--model_fix_flag',
                        type=int,
                        default=0,
                        help='0: No fix, 1:fix.')
    parser.add_argument('--epoch',
                        type=int,
                        default=200,
                        help='Number of epoch.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='Batch size.')

    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate.')
    parser.add_argument('--alpha',
                        type=float,
                        default=0.005,
                        help='gradient coeffcient,[0,1].')
    parser.add_argument('--ratio',
                        type=float,
                        default=0.05,
                        help='unlearn ratio,[0,1].')
    parser.add_argument('--seed', type=int, default=1, help='random seed.')

    parser.add_argument('--save_flag',
                        type=int,
                        default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    return parser.parse_args()
