# Quickstart

A *learning automaton* [NarendraThathachar1989](@cite) is an entity which can direct a number of (probabilistically selected) *actions* to a *learning environment*. In turn, the environment either *rewards* or *punishes* the action chosen by the automaton. The automaton observes the environment's response and adjusts the action probabilities accordingly: actions which are rewarded have their probabilities boosted, while actions which are punished have their probabilities reduced.

!!! info

    The above description is a simplification in many ways. See the [Theory](@ref) 
    section for a more rigorous treatment.

    The term "learning automaton" appears mostly in the engineering literature,
    while in the psychological literature the term "learner" is more common.
    We use both interchangeably.

For a simple example, suppose we have a learning automaton employing the *linear reward--penalty* (LRP) learning scheme [BushMosteller1955](@cite) and three (3) actions. Suppose the learning environment is such that it punishes the first action with probability 0.1, the second action with probability 0.4, and the third action with probability 0.5. How will the learning automaton's action probabilities change as it interacts with this environment?

In LearningAutomata.jl we can implement the above information as follows:

```@example quick
using LearningAutomata

myLearner = LRPLearner(3, 0.005)

myEnvironment = SRE(0.1, 0.4, 0.5)

nothing # hide
```

The learner is created by calling one of the constructors of the `LRPLearner` type. This takes two arguments: the number of actions, and a learning rate parameter which controls how large adjustments the automaton makes to its action probabilities. The environment's type is `SRE`; the environment is constructed by supplying the penalty probabilities to the constructor. The automaton's action probabilities are initially set to the *maximum-entropy state* (1/3, 1/3, 1/3) by default (this means that the learner has no *a priori* information about the learning problem).

To carry out a simulation, we simply need to call a function called `interact!` on the above-defined entities, and provide the number of learning events we wish to simulate as a third function argument:

```@example quick
history = interact!(myLearner, myEnvironment, 10_000)
```

The output is a matrix with 3 columns (one per action) and 10,000 rows (one for each time point). This can be fed into Plots.plot directly in order to obtain a visualization of the learning trajectory:

```@example quick
using Plots

plot(history, label=["action 1" "action 2" "action 3"])
ylims!(0.0, 1.0)

savefig("quickhistory.png") # hide

nothing # hide
```

![](quickhistory.png)

Theoretically it is known that, given this learning environment, the expected values of the action probabilities tend to the following (approximate) values:

- action 1: 0.69
- action 2: 0.17
- action 3: 0.14

We can add these limits to the plot to verify that, after an initial transient, the learner indeed converges on the theoretically predicted values:

```@example quick
hline!([0.69, 0.17, 0.14], color=:black, linestyle=:dash, label=nothing)

savefig("quickhistory2.png") # hide

nothing # hide
```

![](quickhistory2.png)

To learn more about this, consult the [Theory](@ref) section. To see what LearningAutomata.jl can do in more interesting situations, we suggest working through the Guide, starting with [Learning in simple environments](@ref).



