
import os
import sys
import configparser


if __name__ == "__main__":
    experiment_configs = './configs' 
    out_path = './experiment-results'
    arr_configs_pths = os.listdir(experiment_configs)

    for idx, c_name in enumerate(arr_configs_pths):
        print(f"Running {idx+1} of {len(arr_configs_pths)} configs: {c_name}...")
        c_path = os.path.join(experiment_configs, c_name)
        for idx in range(1, 4): 
            if not c_name.endswith('.ini'): 
                print(f"{c_name} isn't a config, babe")
                continue
            config = configparser.ConfigParser()
            config.read(c_path) # can i reuse it?
            model_name = config.get('model', 'model_name')
            log_pth = os.path.join(out_path, model_name)
            
            sys_call = f'python trainer.py --config {c_path} > {log_pth}_run_{idx}.txt'
            os.system(sys_call)
            
