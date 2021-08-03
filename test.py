from aggregators import FedAvgAggregator, MedianAggregator, TrimmedMeanAggregator, KrumAggregator, MultiKrumAggregator
from config import get_model, load_data
from data_attacker import NoDataAttacker, LabelAttacker, NoiseMutator, DeleteMutator, UnbalanceMutator, OverlapMutator
from federated import FedTester

fed_tester = FedTester(
    get_model,
    load_data(),
    FedAvgAggregator(),
    # OverlapMutator(0.3, 1)
)

fed_tester.perform_fed_training(1000)
