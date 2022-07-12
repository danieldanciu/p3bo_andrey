Welcome to Cradle, Ena!

Your mission, should you choose to accept it, is to ... (drum rolls) implement the P3BO algorithm!

Most of the code that you write will go into [p3bo.py](p3bo.py). In order to easily test your implementation we  
provide 3 optimizers to run black box optimization on:
  - [Adalead](flexs/optimizers/adalead.py)
  - [Genetic](flexs/optimizers/genetic_algorithm.py)
  - [Random]((flexs/optimizers/random.py))

The model (or landscape or ground truth) that you are optimizing against is going to be a fake one (a mock) so that 
we avoid running costly/hard-to-setup models. The fake optimizer represents the Levensthein distance from some 
random target sequence that is considered to be the unique optimum solution (so the further you are from the optimum
the higher the value, which means that you have a minimization problem).

The goal is to implement P3BO and that starting with some sequences in the vicinity of the target sequence (the optimum)
the P3BO ensemble converges to the desired value. 

Have fun!