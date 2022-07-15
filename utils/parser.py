import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # For Basic
    parser.add_argument('--name', default = 'main', help = '')

    # For Loader
    parser.add_argument('--train_samples', type = int, default = 100000, help = '')
    parser.add_argument('--valid_samples', type = int, default = 1000, help = '')
    parser.add_argument('--test_samples' , type = int, default = 1000, help = '')

    parser.add_argument('--val_min', type = int, default = 0, help = '')
    parser.add_argument('--val_max', type = int, default = 100, help = '')
    parser.add_argument('--num_min', type = int, default = 5, help = '')
    parser.add_argument('--num_max', type = int, default = 10, help = '')

    # For Module
    parser.add_argument('--dropout', type = float, default = 0.5, help = '')

    parser.add_argument('--emb_dim', type = int, default = 8, help = '')
    parser.add_argument('--hid_dim', type = int, default = 8, help = '')

    parser.add_argument('--num_layers', type = int, default = 1, help = '')     # 若为 2，会下降
    parser.add_argument('--num_directions', type = int, default = 1, help = '') # 若为 2，会下降

    parser.add_argument('--is_coverage', action = 'store_true', help = '')      # 不能开始就用，要等收敛再用
    parser.add_argument('--st_coverage', type = int, default = 31, help = '')   # 这里指定开始使用的迭代轮数

    # For Train
    parser.add_argument('--batch_size', type = int, default = 256, help = '')
    parser.add_argument('--num_epochs', type = int, default = 35, help = '')

    parser.add_argument('--lr', type = float, default = 1e-3, help = '')
    parser.add_argument('--wd', type = float, default = 1e-5, help = '')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    option = parser.parse_args()
