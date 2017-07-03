import train
import sys
import os

if __name__ == '__main__':
    # select which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    if len(sys.argv) < 2:
        print('Usage:')
        print('python main.py -Dataset -CnnStructure -Seed -RestoreModel(optional)')
        print('Dataset: 1. cuhk03   2. cuhk01   3. viper')
        print('CnnStructure: 1. cnn-i  2. cnn-iv  3. cnn-ic  4. cnn-fc-ic  5. cnn-frw-ic')
        print('Seed: [1-20] for cuhk03, [1-10] for cuhk01 and viper')
        print('Example1:  python main.py -cuhk03 -cnn-ic -1')
        print('Example2:  python main.py -cuhk01 -cnn-ic -1 -../model/cnn-ic.ckpt-25001')
    else:
        dataset = sys.argv[1][1:]
        cnn_structure = sys.argv[2][1:]
        current_seed = int(sys.argv[3][1:])
        if len(sys.argv) > 4:
            restore_model = sys.argv[4][1:]
            train.train(dataset, cnn_structure, current_seed, restore_model=restore_model)
        else:
            train.train(dataset, cnn_structure, current_seed)
