import numpy as np
from document_restorer.operations.util import find_max
from document_restorer.operations.util import *


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
        max_cell = find_max(values)
        i, j = max_cell[0], max_cell[1]
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


def find_most_probable_sequence(values):
    sequence, max_probability = [], 0.00
    for n in range(0, values.shape[0]):
        s, p = find_sequence_and_probability(values, n)
        # print str(p) + ' ' + str(s)
        if p > max_probability:
            sequence = s
            max_probability = p
    return sequence


def find_sequence_and_probability(_values, first_fragment):
    values = np.copy(_values)
    priori_probability = 1.00 - float(np.sum(values[:, first_fragment])) / float(np.sum(values))
    sequence = [first_fragment]
    values[:, first_fragment] = np.full((values.shape[0], 1), np.nan)
    current_fragment = first_fragment
    probability = priori_probability
    while len(sequence) < values.shape[0]:
        row = np.copy(values[current_fragment])
        max_index = find_max(row)
        next_fragment = max_index[0] if len(row.shape) > 1 else max_index
        sequence.append(next_fragment)
        row_sum = np.nansum(row)
        probability *= (float(row[max_index]) / float(row_sum))
        values[:, next_fragment] = np.full((values.shape[0], 1), np.nan)
        current_fragment = next_fragment
    return sequence, probability


def restore_document(sequence, connections):
    restored_connections = []
    height = 0
    f0 = None
    for f1 in sequence:
        if f0 is not None:
            connection = connections['{0}-{1}'.format(f0, f1)]
            restored_connections.append(connection)
            height += connection.stuck_fragments.shape[0]
        f0 = f1
    restored_document = np.zeros(
        [height, restored_connections[0].stuck_fragments.shape[1]], restored_connections[0].stuck_fragments.dtype)
    x, y = 0, 0
    for connection in restored_connections:
        copy_to(connection.stuck_fragments, restored_document, x, y)
        x += connection.offset[0]
        y += connection.stuck_fragments.shape[0]
    return restored_document
