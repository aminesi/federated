import json

datasets = ['cifar']
non_iids = [0.4]
attacks = ['noise-data', 'overlap-data', 'delete-data', 'unbalance-data']
attack_fractions = [0.3]
aggregators = ['fed-avg']

conf = {'num-rounds': 1000}

i = 0


def show(i):
    for aggregator in aggregators:
        i += 1
        conf['aggregator'] = aggregator
        print(conf)
        with open('../configs/prem/config-{}.json'.format(i), 'w') as file:
            json.dump(conf, file)
            file.close()
    return i


for dataset in datasets:
    conf['dataset'] = dataset
    for non_iid in non_iids:
        conf['non-iid-deg'] = non_iid
        for attack in attacks:
            conf['attack'] = attack
            if attack:
                for attack_fraction in attack_fractions:
                    conf['attack-fraction'] = attack_fraction
                    if 'noise' in attack:
                        params = {'sigma_multiplier': [0.1, 0.5, 1]}
                    else:
                        params = {attack.replace('-data', '') + '_percentage': [0.25, 0.5, 0.75]}
                    key = list(params.keys())[0]
                    for p in params[key]:
                        conf[key] = p
                        i = show(i)
                    del conf[key]
            else:
                del conf['attack']
                if 'attack-fraction' in conf:
                    del conf['attack-fraction']
                i = show(i)

print(i)
