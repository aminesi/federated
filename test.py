from attacks.data_attacker import LabelAttacker, NoiseMutator, OverlapMutator, DeleteMutator, UnbalanceMutator
from fed.aggregators import MedianAggregator, FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator
from config import get_model, load_data, get_num_round, get_param, throw_conf_error
from fed.federated import FedTester
from attacks.model_attacker import SignFlipModelAttacker, BackdoorAttack, RandomModelAttacker
from utils.util import ADNIDataset, Dataset

result = load_data()

dataset = None
if isinstance(result, str):
    dataset = ADNIDataset(result)
else:
    dataset = Dataset(*result)


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
        attack_map['data_attacker'] = NoiseMutator(fraction)
    elif attack == 'overlap-data':
        attack_map['data_attacker'] = OverlapMutator(fraction)
    elif attack == 'delete-data':
        attack_map['data_attacker'] = DeleteMutator(fraction)
    elif attack == 'unbalance-data':
        attack_map['data_attacker'] = UnbalanceMutator(fraction)
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

fed_tester.perform_fed_training(get_num_round())
