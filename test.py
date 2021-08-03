from attacks.data_attacker import LabelAttacker
from fed.aggregators import MedianAggregator
from config import get_model, load_data, get_num_round
from fed.federated import FedTester
from attacks.model_attacker import SignFlipModelAttacker

fed_tester = FedTester(
    get_model,
    load_data(),
    MedianAggregator(),
    data_attacker=LabelAttacker(0.6)
    # model_attacker=SignFlipModelAttacker(0.3, 2)
)

fed_tester.perform_fed_training(get_num_round())
