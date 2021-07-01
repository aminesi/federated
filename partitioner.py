from typing import List, Union
import numpy as np

NUM_CLIENTS = 100


def map_deg(non_iid_deg: float, class_count: int) -> int:
    return int(np.rint((1 - non_iid_deg) * (class_count - 1) + 1))


def get_partitioned_indices(labels: Union[List[int], np.ndarray], non_iid_deg: float, num_clients=NUM_CLIENTS) \
        -> List[np.ndarray]:
    labels = np.array(labels)
    if labels.ndim > 1:
        raise AttributeError('labels should be a flat array')
    class_count = len(np.unique(labels))
    if non_iid_deg < 0 or non_iid_deg > 1:
        raise AttributeError('non_iid_deg should be between 0 and 1')

    iid_deg = map_deg(non_iid_deg, class_count)
    chunk_count = iid_deg * num_clients
    indices_chunks = np.array_split(labels.argsort(), chunk_count)
    np.random.shuffle(indices_chunks)
    client_data_arr = []
    for i in range(num_clients):
        client_chunks = indices_chunks[i * iid_deg:(i + 1) * iid_deg]
        client_data_arr.append(np.concatenate(client_chunks))

    return client_data_arr

