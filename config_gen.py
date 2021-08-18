import json

file_name_base = "config_set/config_"
base_json = {
  "num-rounds": 100,
  "dataset": "mnist",
  "non-iid-deg": 0,
  "aggregator": "fed-avg",
  "attack": "random-update",
  "attack-fraction": 0.5
}

#  'random-update', attack
possible_configs = {
    'dataset': {'mnist', 'cifar'},
    'aggregator': {'fed-avg', 'median', 'trimmed-mean', 'krum'},
    'attack': {'label-flip', 'noise-data', 'overlap-data', 'delete-data', 'unbalance-data','random-update',
               'sign-flip', 'backdoor'},
    'attack-fraction': 'float between 0 and 1',
    'non-iid-deg': 'float between 0 and 1',
    'num-rounds': 'integer value'
}

possible_attacks = {'label-flip', 'noise-data', 'overlap-data', 'delete-data', 'unbalance-data'}

count = 27
for aggregator_way in possible_configs["aggregator"]:
    base_json["aggregator"] = aggregator_way

    for attack_way in possible_configs["attack"]:
        base_json["attack"] = attack_way 
        for non_iid_param in [0,0.4,0.7]:
            for attack_frac in [0.1,0.3]:
                base_json["attack-fraction"] = attack_frac  
                file_name = "%s_%s.json" %(file_name_base, count)
                with open(file_name,"w") as fp:
                    json.dump(base_json,fp)
                    count+=1
print(count)
