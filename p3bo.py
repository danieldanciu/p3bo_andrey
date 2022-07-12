""" This is the file that contains the main P3BO method that you are supposed to implement. """
from typing import List

import flexs.explorer
from flexs.optimizers.random import Random
from flexs.optimizers.adalead import Adalead
from flexs.optimizers.genetic_algorithm import GeneticAlgorithm


class P3bo:
    def __init__(self, explorers: List[flexs.explorer.Explorer]):
        self.explorers = explorers





def main():
    adalead = ...
    ga = ...
    random = ...
    p3bo = P3bo([random, ga, adalead])

    p3bo.optimize()

if __name__ == "__main__":
    main()
