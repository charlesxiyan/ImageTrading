from __rd__ import *

class InitialSetting():
    def __init__(self, yaml_dir):
        # Note that the input of parameters should be saved in a yaml_file
        self.yaml = yaml_dir
        self.read_yaml()
    
    def preload(self):
        # Make sure the directory is correct for model saving after train-validate, i.e. the github project filefolder
        dir_list = os.listdir('./')
        if 'models' not in (dir_list):
            os.system('mkdir models')
        if self.config.TRAIN.MODEL_SAVE_FILE.split('/')[-2] not in os.listdir('./models/'):
            os.system('cd models && mkdir {}'.format(self.config.TRAIN.MODEL_SAVE_FILE.split('/')[-2]))
        
        # Make sure the directory is correct for log (train/validate loss/accuracy) saving after train-validate
        if 'logs' not in dir_list:
            os.system('mkdir logs')
        if self.config.TRAIN.LOG_SAVE_FILE.split('/')[-2] not in os.listdir('./logs/'):
            os.system('cd logs && mkdir {}'.format(self.config.TRAIN.LOG_SAVE_FILE.split('/')[-2]))
    
    def read_yaml(self):
        # Extract info from yaml_file and turn it to a namedtuple
        with open(self.yaml, 'r') as file:
            self.setting = yaml.safe_load(file)
        
        # Pre-setting Stage
        assert (type(self.setting)) is dict, 'Target is not an YAML File.'
        
        # Now extract info from YAML file and create a namedtuple object named after 'root'
        self.config = self.yaml_2_namedtuple('root', self.setting)
        
        self.preload()
    
    def yaml_2_namedtuple(self, name, value):
        obj_type = type(value)
        if obj_type is dict:
            # The first step of converting yaml to namedtuple will go thru this if-statement for sure, since yaml itself is a dictionary.
            # Thus, create a namedtuple data structure named after the variable 'name'
            nt = namedtuple(name, value.keys())
            # Thru DFS to keep creating sub-namedtuple if dict type data is still contained
            to_nt = nt(*[self.yaml_2_namedtuple(sub_name, sub_value) for sub_name, sub_value in value.items()])
        elif obj_type is list:
            # If there exists an item in the dict, say 'Sizes': [10, 100, 1000], then the value return is still a list.
            # However, in case of 'SomeKey': [9, 4, [2, 3, 5], {'a': 'Alice', 'b': 'Bill'}], an iteration of elements within
            to_nt = [self.yaml_2_namedtuple(f'{name}_{i}', value[i]) for i in range(len(value))]
        else:
            # The simplest case for deterministic values, say strings and numbers.
            to_nt = value
        return to_nt
    
    def replace_yaml(self, replace_dict, replaced_dict = None):
        assert isinstance(replace_dict, dict), f'Incorrect input: a dictionary is expected.'
        if replaced_dict == None:
            replaced_dict = self.setting
        
        for key, value in replace_dict.items():
            if key in replaced_dict.keys():
                if isinstance(value, dict):
                    self.replace_yaml(replace_dict = value, replaced_dict = replaced_dict[key])
                else:
                    replaced_dict[key] = value
          
        self.setting = replaced_dict
                    
    def update_config(self, replace_dict):
        # Update the dictionary from yaml and then update namedtuple
        # The reason why we do not directly update namedtuple is that namedtuple object cannot be changed directly
        self.replace_yaml(replace_dict)
        # Update config and check relevant directories if they need creating
        self.config = self.yaml_2_namedtuple('root', self.setting)
        
        self.preload()

        