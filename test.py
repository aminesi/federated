from attacks.data_attacker import LabelAttacker
from fed.aggregators import MedianAggregator, FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator
from config import get_model, load_data, get_num_round
from fed.federated import FedTester
from attacks.model_attacker import SignFlipModelAttacker, BackdoorAttack, RandomModelAttacker
from utils.util import ADNIDataset, Dataset

result = load_data()

dataset = None
if isinstance(result, str):
    dataset = ADNIDataset(result)
else:
    dataset = Dataset(*result)

fed_tester = FedTester(
    get_model,
    dataset,
    MedianAggregator(),
    # data_attacker=LabelAttacker(0.5)
    model_attacker=RandomModelAttacker(.5, 1)
)

fed_tester.perform_fed_training(get_num_round())
