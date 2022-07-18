""" This is the file that contains the main P3BO method that you are supposed to implement. """
from email.mime import base
from typing import List
import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import flexs.explorer
from flexs.models.levensthein import LevenstheinLandscape
from flexs.models.noisy_abstract_model import NoisyAbstractModel
from flexs.optimizers.random import Random
from flexs.optimizers.adalead import Adalead
from flexs.optimizers.genetic_algorithm import GeneticAlgorithm

protein_alphabet = 'ACDEFGHIKLMNPQRSTVWY'

optimal_sequence = 'MKYTKVMRYQIIKPLNAEWDELGMVLRDIQKETRAALNKTIQLCWEYQGFSADYKQIHGQYPKPKDVLGYTSMHGYAYDRLKNEFSKIASSNLSQTIKRAVDKWNSDLKEILRGDRSIPNFRKDCPIDIVKQSTKIQKCNDGYVLSLGLINREYKNELGRKNGVFDVLIKANDKTQQTILERIINGDYTYTASQIINHKNKWFINLTYQFETKETALDPNNVMGVDLGIVYPVYIAFNNSLHRYHIKGGEIERFRRQVEKRKRELLNQGKYCGDGRKGHGYATRTKSIESISDKIARFRDTCNHKYSRFIVDMALKHNCGIIQMEDLTGISKESTFLKNWTYYDLQQKIEYKAREAGIQVIKIEPQYTSQRCSKCGYIDKENRQEQATFKCIECGFKTNADYNAARNIAIPNIDKIIRKTLKMQ'

num_iterations = 20
        
decay_factor = 0.25
softmax_t = 1.0
population_per_explorer = 100

def validate_optimal_sequence():
    for ch in optimal_sequence:
        if ch not in protein_alphabet:
            return False
    return True

def get_starting_sequence(base_sequence: str, identity_percent: float) -> str:
    """ This function returns a sequence that is identity_percent identical to the given base sequence """
    seq_length: int = len(base_sequence)
    number_of_mutations: int = seq_length * (100 - identity_percent) // 100
    indices_to_mutate = random.sample(range(seq_length),number_of_mutations)
    mutated_sequence = list(base_sequence)
    for index in indices_to_mutate:
        mutated_sequence[index] = random.choice(protein_alphabet)
    return ''.join(mutated_sequence)


class P3bo:
    def __init__(self, explorers: List[flexs.explorer.Explorer]):
        self.explorers = explorers

    def hack_create_dataframe(self):
        first_explorer = self.explorers[0]
        model = first_explorer.model
        starting_sequence = first_explorer.starting_sequence
        starting_sequence_fitness = model.get_fitness([starting_sequence])
        d = {'sequence': [starting_sequence], 'true_score': [starting_sequence_fitness]}
        return pd.DataFrame(data=d)

    def optimize(self):
        """
        This is the function that you need to implement (including adding the necessary parameters)

        The P3BO population will consist of the three algorithms: adalead, random and genetic (they are in the
        optimizers directory). Each of these algorithms exposes a propose_sequences() method and a fit() method.

        You don't need to implement the adaptive variant of P3BO.
        """
        model = LevenstheinLandscape(optimal_sequence)
        num_explorers = len(self.explorers)
        population_size = population_per_explorer * num_explorers
        credit_scores = [0] * num_explorers
        df = self.hack_create_dataframe()
        cached_sequences = list()
        cached_scores = list()
        f_max = float("-inf")

        rewards_per_iteration = np.zeros([0, num_explorers])

        for t in tqdm(range(1, num_iterations + 1)):
            proposed_sequences = list(map(lambda explorer: explorer.propose_sequences(df), self.explorers))
            for explorer_sequences in proposed_sequences:
                # TODO: filter out similar sequences
                cached_sequences += explorer_sequences[0].tolist()
                cached_scores += model.get_fitness(explorer_sequences[0]).tolist()
                data = {'sequence': cached_sequences, 'true_score': cached_scores}
                df = pd.DataFrame(data=data)

            max_rewards = list(map(lambda seq: max(model.get_fitness(seq[0])), proposed_sequences))
            rewards_per_iteration = np.append(rewards_per_iteration, [max_rewards], axis=0)
            f_max = max(f_max, max(max_rewards))
            relative_rewards = list(map(lambda reward: (reward - f_max)/f_max, max_rewards))
            decay = decay_factor ** t
            for i in range(num_explorers):
                credit_scores[i] += relative_rewards[i] * decay
            decayed_credit_scores = list(map(lambda s: math.exp(s/softmax_t), credit_scores))
            decayed_credit_score_sum = sum(decayed_credit_scores)
            relative_population_sizes = list(map(lambda decayed_credit_score: decayed_credit_score / decayed_credit_score_sum, decayed_credit_scores))
            for i in range(num_explorers):
                self.explorers[i].model_queries_per_batch = int(relative_population_sizes[i] * population_size)
            for explorer in self.explorers:
                explorer.fit(cached_sequences, cached_scores)

        x_axis = np.arange(num_iterations)
        plt.plot(x_axis, rewards_per_iteration)
        plt.xlabel("Iteration") 
        plt.ylabel("Fitness function")
        plt.legend(list(map(lambda explorer: explorer.name, self.explorers)))
        plt.show()
        
    def optimize_naive(self):
        model = self.explorers[0].model
        best_seq = ""
        best_fitness = 0
        best_explorer = None
        for explorer in self.explorers:
            print("Starting explorer: ", explorer)
            proposed_sequences = explorer.propose_sequences(self.hack_create_dataframe())
            fitnesses = model.get_fitness(proposed_sequences[0])
            for i in range(len(fitnesses)):
                if fitnesses[i] > best_fitness:
                    best_seq = proposed_sequences[0][i]
                    best_fitness = fitnesses[i]
                    best_explorer = explorer
            print("Best fitness for this explorer: ", best_fitness)
        print(best_fitness)
        print(best_seq)
        print(best_explorer)


def main():
    # create a naive/mock model that simply computes the distance from the target optimum
    model = NoisyAbstractModel(LevenstheinLandscape(optimal_sequence), signal_strength=1)

    starting_sequence = get_starting_sequence(optimal_sequence, 80)  # get a sequence 80% identical to the optimal
    adalead = Adalead(model=model, rounds=10, sequences_batch_size=10, model_queries_per_batch=population_per_explorer,
                      starting_sequence=starting_sequence, alphabet=protein_alphabet)
    ga = GeneticAlgorithm(model=model, rounds=10, sequences_batch_size=10, model_queries_per_batch=population_per_explorer,
                          starting_sequence=starting_sequence, alphabet=protein_alphabet, population_size=100,
                          parent_selection_strategy='top-proportion', children_proportion=0.5,
                          parent_selection_proportion=0.5)
    random_algorithm = Random(model=model, rounds=10, sequences_batch_size=10, model_queries_per_batch=population_per_explorer,
                    starting_sequence=starting_sequence, alphabet=protein_alphabet)

    p3bo = P3bo([random_algorithm, ga, adalead])

    # that's the method you have to implement
    # p3bo.optimize_naive()
    p3bo.optimize()


if __name__ == "__main__":
    main()
