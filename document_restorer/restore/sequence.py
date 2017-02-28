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


# todo doesn't consider priori probability of first fragment choice
def find_most_probable_sequence(values):
    sequence, max_probability = [], 0.00
    for n in range(0, values.shape[0]):
        s, p = find_sequence_and_probability(values, n)
        if p > max_probability:
            sequence = s
            max_probability = p
    return sequence


def find_sequence_and_probability(_values, first_fragment):
    values = np.copy(_values)
    sequence = [first_fragment]
    values[:, first_fragment] = np.full((values.shape[0], 1), np.nan)
    current_fragment = first_fragment
    probability = 1.00
    while len(sequence) < values.shape[0]:
        row = np.copy(values[current_fragment])
        min_index = find_min(row)
        next_fragment = min_index[0] if len(row.shape) > 1 else min_index
        sequence.append(next_fragment)
        row_sum = np.nansum(row)
        probability *= (float(row[min_index]) / float(row_sum))
        values[:, next_fragment] = np.full((values.shape[0], 1), np.nan)
        current_fragment = next_fragment
    return sequence, probability
