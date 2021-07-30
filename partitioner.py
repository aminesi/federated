from typing import List, Union
import numpy as np

from constants import NUM_CLIENTS


def map_deg(non_iid_deg: float, class_count: int) -> int:
    return int(np.rint((1 - non_iid_deg) * (class_count - 1) + 1))


def get_indices_by_class(labels):
    return [np.where(labels == label)[0] for label in np.unique(labels)]


def get_partitioned_indices(labels: Union[List[int], np.ndarray], non_iid_deg: float, num_clients=NUM_CLIENTS) \
        -> List[np.ndarray]:
    labels = np.array(labels)
    if labels.ndim > 1:
        raise AttributeError('labels should be a flat array')
    class_count = len(np.unique(labels))
    if non_iid_deg < 0 or non_iid_deg > 1:
        raise AttributeError('non_iid_deg should be between 0 and 1')

    grouped_indices = get_indices_by_class(labels)

    iid_deg = map_deg(non_iid_deg, class_count)
    chunk_count = iid_deg * num_clients
    per_class_chunk_count = chunk_count // class_count

    grouped_chunks = []
    for per_class_indices in grouped_indices:
        np.random.shuffle(per_class_indices)
        chunks = np.array_split(per_class_indices, per_class_chunk_count)
        grouped_chunks.append(chunks)

    pool = [i for i in range(10)]
    remainder = []
    new_i = 0
    client_data_arr = []
    for i in range(num_clients):
        pool = list(filter(lambda class_index: grouped_chunks[class_index], pool))
        if len(pool) < iid_deg:
            if not remainder:
                new_i = i
                for j in pool:
                    remainder += grouped_chunks[j]
                np.random.shuffle(remainder)
            i = i - new_i
            client_chunks = remainder[i * iid_deg:(i + 1) * iid_deg]
            client_data_arr.append(np.concatenate(client_chunks))
        else:
            client_chunks = []
            for choice in np.random.choice(pool, iid_deg, False):
                client_chunks.append(grouped_chunks[choice].pop())
            client_data_arr.append(np.concatenate(client_chunks))
    np.random.shuffle(client_data_arr)
    return client_data_arr
