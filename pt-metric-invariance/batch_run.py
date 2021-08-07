
import os
import sys
import configparser
import argparse

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--runs', type=int, default=3)
    # parser.add_argument('--out', type=str, required=True, default='./experiment-results')
    # parser.add_argument('--configs', type=str, default='./configs')
    # parser.add_argument('-f', type=bool, value=default=True)
    # input_args = parser.parse_args()


    # # if os.path.exists(input_args.out): 
    #     # if 
    # assert os.path.exists(input_args.configs)
    experiment_configs = './configs-dev' #'./configs folder to run' 
    out_path = './batch-metric-analysis' #'./experiment results folder name'
    arr_configs_pths = os.listdir(experiment_configs)

    for idx, c_name in enumerate(arr_configs_pths):
        print(f"Running {idx+1} of {len(arr_configs_pths)} configs: {c_name}...")
        c_path = os.path.join(experiment_configs, c_name)
        for idx in range(1, 2): #TODO change
            if not c_name.endswith('.ini'): 
                print(f"{c_name} isn't a config, babe")
                continue
            config = configparser.ConfigParser()
            config.read(c_path) # can i reuse it?
            model_name = config.get('model', 'model_name')
            log_pth = os.path.join(out_path, model_name)
            
            sys_call = f'python trainer.py --device 1 --config {c_path} > {log_pth}_run_{idx}.txt'
            os.system(sys_call)
            
