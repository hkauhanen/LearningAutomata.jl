# Guide

## Creating learners

The following creates a two-action linear reward-penalty learner with global
learning rate 0.001.

```@repl
using LearningAutomata
alice = LRPLearner(2, 0.001)
```

The second constructor argument may also be a vector. Here we create a 
three-action learner with asymmetric learning rates for the different actions:

```@repl
using LearningAutomata # hide
bob = LRPLearner(3, [0.001, 0.1, 0.02])
```

The remaining fields are set using keyword arguments in the constructor. 
For instance, to force a different learning rate vector for penalties:

```@repl
using LearningAutomata # hide
carla = LRPLearner(3, 0.001, b = [0.1, 0.2, 0.01])
```


## Simulations in a stationary random environment

```@repl
using LearningAutomata # hide
using Plots
ENV["GKSwstype"] = "100" # hide
alice = LRPLearner(3, 0.01)
sre = [0.4, 0.2, 0.1]
simulate!(alice, 1000, sre) |> plot
savefig("simulation.svg"); nothing # hide
```

![](simulation.svg)


## Reconfiguring learners


## Games
