import numpy as np
from document_restorer.operations.util import find_min


class FragmentsSequenceBuilder(object):

    def __init__(self):
        self.chains = []

    # todo think about circuit connection recognition
    def link(self, f1, f2):
        chains_with_f1 = [chain for chain in self.chains if f1 in chain]
        chains_with_f2 = [chain for chain in self.chains if f2 in chain]
        if len(chains_with_f1) > 0 and len(chains_with_f2) > 0:
            if chains_with_f1[0] == chains_with_f2[0]:
                return False
            if not(chains_with_f1[0][-1] == f1 and chains_with_f2[0][0] == f2):
                raise Exception('internal sequence builder error #1')
            new_chain = chains_with_f1[0] + chains_with_f2[0]
            self.chains.remove(chains_with_f1[0])
            self.chains.remove(chains_with_f2[0])
            self.chains.append(new_chain)
        elif len(chains_with_f1) > 0:
            chains_with_f1[0].insert(chains_with_f1[0].index(f1) + 1, f2)
        elif len(chains_with_f2) > 0:
            chains_with_f2[0].insert(chains_with_f2[0].index(f2), f1)
        else:
            self.chains.append([f1, f2])
        return True

    def build(self):
        if len(self.chains) > 1:
            raise Exception('internal sequence builder error #2')
        return self.chains[0]


def find_sequence(_values):
    values = np.copy(_values)
    connections_added = 0
    sequence_builder = FragmentsSequenceBuilder()
    while connections_added < values.shape[0] - 1 and np.count_nonzero(~np.isnan(values)) > 0:
        min_ind = find_min(values)
        i, j = min_ind[0], min_ind[1]
        if i == j:
            values[i, j] = np.nan
            continue
        if sequence_builder.link(i, j):
            # print '{0}-{1}'.format(i, j)
            values[i, :] = np.full((values.shape[0], 1), np.nan)
            values[:, j] = np.full((values.shape[1], 1), np.nan)
            values[j, i] = np.nan
            connections_added += 1
        else:
            values[i, j] = np.nan
    return sequence_builder.build()


def find_sequence_with_known_first_fragment(values, f0):
    sequence, restricted_columns = [], []
    next_index = f0
    while True:
        sequence.append(next_index)
        restricted_columns.append(next_index)
        if len(sequence) < values.shape[0]:
            row = np.copy(values[next_index])
            max_value = np.max(row) + 1.00
            row[restricted_columns] = max_value
            next_index = np.argmin(row)
        else:
            break
    return sequence
