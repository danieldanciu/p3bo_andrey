from typing import List, Any

import editdistance
import flexs.landscape
import numpy as np


class LevenstheinLandscape(flexs.landscape.Landscape):
    """ A simple landscape that returns the Levensthein distance from the optimum sequence as the fitness value """

    def __init__(self, target_sequence):
        super().__init__("Levensthein")
        self.target_sequence = target_sequence

    def _fitness_function(self, sequences: flexs.model.SEQUENCES_TYPE) -> np.ndarray:
        result = []
        ts_len = float(len(self.target_sequence))
        for seq in sequences:
            # The fitness function ranges from 0 (completely different sequence) to 1 (equal to original sequence)
            result.append((ts_len - editdistance.eval(seq, self.target_sequence))/ts_len)
        return np.array(result)

    def train(self, sequences: flexs.model.SEQUENCES_TYPE, labels: List[Any]):
        """ The Levensthein model cannot be trained """
        pass
