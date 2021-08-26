import json

datasets = ['cifar']
non_iids = [0, 0.4, 0.7]
attacks = [None, 'label-flip', 'random-update', 'sign-flip', 'backdoor']
attack_fractions = [0.1, 0.3, 0.5]
aggregators = ['fed-avg', 'krum', 'median', 'trimmed-mean']

conf = {'num-rounds': 1000}

i = 0


def show(i):
    for aggregator in aggregators:
        if aggregator == 'fed-avg' and 'attack' in conf and conf['attack'] != 'label-flip':
            continue
        if aggregator != 'krum' and 'attack' in conf and conf['attack'] == 'backdoor':
            continue
        i += 1
        conf['aggregator'] = aggregator
        print(conf)
        with open('../runs/config-{}.json'.format(i), 'w') as file:
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
                    i = show(i)
            else:
                del conf['attack']
                if 'attack-fraction' in conf:
                    del conf['attack-fraction']
                i = show(i)

print(i)
