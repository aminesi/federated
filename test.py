from attacks.data_attacker import LabelAttacker, NoiseMutator, OverlapMutator, DeleteMutator, UnbalanceMutator
from fed.aggregators import MedianAggregator, FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator
from config import get_model, load_data, get_num_round, get_param, throw_conf_error, get_result_dir
from fed.federated import FedTester
from attacks.model_attacker import SignFlipModelAttacker, BackdoorAttack, RandomModelAttacker
from utils.util import ADNIDataset, Dataset
import numpy as np

data_loader_result = load_data()

dataset = None
if isinstance(data_loader_result, str):
    dataset = ADNIDataset(data_loader_result)
else:
    dataset = Dataset(*data_loader_result)


def get_aggregator():
    aggregator = get_param('aggregator', 'fed-avg')
    if aggregator == 'fed-avg':
        return FedAvgAggregator()
    if aggregator == 'krum':
        return KrumAggregator()
    if aggregator == 'median':
        return MedianAggregator()
    if aggregator == 'trimmed-mean':
        return TrimmedMeanAggregator()
    if aggregator == 'combine':
        return None
    throw_conf_error('aggregator')


def get_attacks():
    attack_map = {}
    attack = get_param('attack', 'none')
    if attack == 'none':
        return attack_map
    fraction = get_param('attack-fraction')
    if attack == 'label-flip':
        attack_map['data_attacker'] = LabelAttacker(fraction)
    elif attack == 'noise-data':
        attack_map['data_attacker'] = NoiseMutator(fraction, get_param('sigma_multiplier', 1))
    elif attack == 'overlap-data':
        attack_map['data_attacker'] = OverlapMutator(fraction, get_param('overlap_percentage', 0.75))
    elif attack == 'delete-data':
        attack_map['data_attacker'] = DeleteMutator(fraction, get_param('delete_percentage', 0.75))
    elif attack == 'unbalance-data':
        attack_map['data_attacker'] = UnbalanceMutator(fraction, get_param('unbalance_percentage', 0.75))
    elif attack == 'random-update':
        attack_map['model_attacker'] = RandomModelAttacker(fraction, 2)
    elif attack == 'sign-flip':
        attack_map['model_attacker'] = SignFlipModelAttacker(fraction, 10)
    elif attack == 'backdoor':
        attack_map['model_attacker'] = BackdoorAttack(fraction, 1)
    else:
        throw_conf_error('attack')
    return attack_map


fed_tester = FedTester(
    get_model,
    dataset,
    get_aggregator(),
    **get_attacks()
)

results = fed_tester.perform_fed_training(get_num_round())

results_dir = get_result_dir()
for key in results:
    file_path = results_dir + key
    result = results[key]
    if key == 'time':
        with open(file_path, 'w') as file:
            file.write(str(result))
            file.close()
    else:
        np.save(file_path, np.array(result))
