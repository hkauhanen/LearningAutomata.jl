# Agents.jl integration

```@setup agents
using Random
Random.seed!(123)
```

For simulations more complex than those outlined in [Learning in simple environments](@ref), it is useful to integrate LearningAutomata.jl with the powerful [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) package. The recommended way of doing this is via [composition](https://en.wikipedia.org/wiki/Composition_over_inheritance): a composite type intended to be used as an agent type as part of an Agents.jl simulation will contain e.g. an `LRPLearner` type as a field.

!!! note

    If you are unfamiliar with Agents.jl, it is advisable to work through their
    [tutorial](https://juliadynamics.github.io/Agents.jl/stable/tutorial/)
    before proceeding.

To illustrate this, we will here set up a simulation with 100 learners, each learner occupying one node in a 10-by-10 lattice and interacting with its nearest neighbours only. This example is quite contrived, and does not necessarily do anything of inherent interest; yet it serves to illustrate the basic concepts and mechanisms. Adopting the code for other kinds of spaces provided by Agents.jl, such as graphs, should be straightforward.


## Defining the agent type

To begin, we load Agents.jl and define a new composite type called `myLearner` which contains an `LRPLearner` as a field which we name `brain` by convention. We also add a field called `age::Int` which tracks the number of interactions the agent has taken part in, as well as `max_age::Int` which records the agent's maximum age. These variables will be used to model aging and to remove old agents from the population.

```@example agents
using Agents
using LearningAutomata

@agent struct myLearner(GridAgent{2})
  brain::LRPLearner
  age::Int
  max_age::Int
end
```

Eventually, we will want to have some social variability in our population of learners; we will do this by drawing each agent's `max_age` from a normal distribution. For this, we set up a couple of such distributions and define a function that returns a random variate which only preserves the integer part (since our agents' ages are integers):

```@example agents
using Distributions

# life expectancy = 100K learning iterations
distro1 = Normal(100_000, 1000)

# life expectancy = 5K learning iterations
distro2 = Normal(5000, 100)

# sampler
get_lifetime(x) = trunc(Int, rand(x))

nothing # hide
```

The lifetime distribution will ultimately be encoded as a model parameter in the agent-based model; see below.


## Defining a stepper function

To define a (discrete-time) simulation in Agents.jl, we need to set up either a model stepping function or an agent stepping function. We opt for a model stepping function and call this function `mystep!`. We assume that it does the following things:

1. Sample an agent at random (the "focal agent")
1. Sample a lattice neighbour of the focal agent
1. Undergo a non-reciprocal interaction between these two agents, in which the focal updates its state
1. Increment the focal agent's age
1. Check whether maximum age has been reached and if so, reset the focal agent's knowledge state to the maximum-entropy state to mimic death followed by the birth of a *tabula rasa* learner

Here is our implementation:

```@example agents
function mystep!(model)
  agent = random_agent(model)
  neighbour = random_nearby_agent(agent, model)

  interact!(neighbour.brain, agent.brain; reciprocal = false)
  
  agent.age += 1

  if agent.age > agent.max_age
    reset!(agent.brain)
    agent.age = 0
    agent.max_age = get_lifetime(model.nd)
  end
end

nothing # hide
```

!!! danger
    
    Note that the arguments to `interact!` are **not** `neighbour` and `agent`, 
    but rather the `LRPLearner`s contained in them!

!!! tip

    The `reset!(x)` function sets learner `x`'s weight vector to the maximum-entropy state ``(1/n, \ldots , 1/n)``, where ``n`` is the number of actions.

## Defining a space and initializing the model

We next define the lattice and instantiate an agent-based model.

```@example agents
dims = (10, 10)
space = GridSpace(dims)

model = StandardABM(myLearner,
                    space,
                    model_step! = mystep!,
                    properties = Dict(:nd => distro1))
```

Next, we add 100 agents to the model:

```@example agents
for i in 1:100
  add_agent_single!(model,
                    LRPLearner(2, 0.01; W = [0.1, 0.9], A = [0.0 0.1; 0.2 0.0]), 
                    0,
                    get_lifetime(model.nd))
end
```

Note the assumptions here: we assume 2 actions, a global learning rate of 0.01, an initial state $\mathbf{W} = (0.1, 0.9)$ for each learner, as well as the following advantage matrix:

$$A = \begin{pmatrix} 0 & 0.1 \\ 0.2 & 0 \end{pmatrix}.$$

(This means that the first action enjoys an advantage over the second action and its weight is thus expected to increase over time -- something to bear in mind in view of the simulations to come.) We also set each new agent's age to zero, and draw a lifetime from the lifetime distribution in `model.nd`.


## Stepping the model and visualizing the population state

Let's run the model for a few steps:

```@example agents
step!(model, 100)

nothing # hide
```

We can visualize the state of the population, for instance by colouring each node of the lattice by the weight on the first action of the agent in that node. To do this, we first define a getter for that first weight:

```@example agents
first_weight(x::myLearner) = x.brain.W[1]

nothing # hide
```

We can now plot:

```@example agents
using CairoMakie

fig, _ = abmplot(model; agent_color = first_weight, agent_size = 20)

save("population.png", fig) # hide
nothing # hide
```

![](population.png)

## Evolution of the mean

In most applications, it is of interest to track the evolution of the population mean -- in the current example, the average weight on the first action computed across the lattice. With Agents.jl, this is easy:

```@example agents
using Statistics

adata = [(first_weight, mean)]

adf, mdf = run!(model, 1_000_000; adata)
```

A simple line plot shows how this mean evolves over time:

```@example agents
using Plots
Plots.plot(adf.time, adf.mean_first_weight, label="mean W1")

savefig("meanweight.png") # hide
nothing # hide
```

![](meanweight.png)

## The effect of model parameters

To illustrate the effect model parameters can have on a simulation like this, suppose we decrease life expectancies so that learners in general fail to converge on the expected weight vector (cf. [Theory](@ref)) -- but keep everything else the same. To use this, we now employ the lifetime distribution `distro2` defined above, which implies a mean life expectancy of only 5000 learning iterations compared to the 100,000 used before:

```@example agents
dims = (10, 10)
space2 = GridSpace(dims)

model2 = StandardABM(myLearner,
                    space2,
                    model_step! = mystep!,
                    properties = Dict(:nd => distro2))

for i in 1:100
  add_agent_single!(model2, 
                    LRPLearner(2, 0.01; W = [0.1, 0.9], A = [0.0 0.1; 0.2 0.0]), 
                    0, 
                    get_lifetime(model2.nd))
end

adf2, mdf2 = run!(model2, 3_000_000; adata)

Plots.plot(adf2.time, adf2.mean_first_weight, label="mean W1")

savefig("meanweight2.png") # hide
nothing # hide
```

![](meanweight2.png)

As a consequence, we obtain oscillatory behaviour at population level.


## A continuous-time simulation

In the above simulations time evolves by discrete ticks. However, in the mathematical literature on learning automata, the establishment of certain important limits assumes that learning is a *Markov jump process*: learning opportunities come to the learner at exponentially arriving times, and between these times the agent's state remains unchanged. This sort of model can be implemented using the `EventQueueABM` model type of Agents.jl.

We first define an agent stepping function. This is basically the same as `mystep!` above, except that we do not sample the focal agent as the event scheduler of `EventQueueABM` does this for us:

```@example agents
function myagentstep!(agent, model)
  neighbour = random_nearby_agent(agent, model)

  interact!(neighbour.brain, agent.brain; reciprocal = false)
  
  agent.age += 1

  if agent.age > agent.max_age
    reset!(agent.brain)
    agent.age = 0
    agent.max_age = get_lifetime(model.nd)
  end
end

nothing # hide
```

The exponentially arriving events in our model are defined by the following `AgentEvent` structure. The action is simply to call the `myagentstep!` function; `propensity` encodes the event rate.

```@example agents
myevent = AgentEvent(; action! = myagentstep!, propensity = 0.001)
nothing # hide
```

We can now proceed to define the space and the model.

!!! warning

    In the following code listing, note the argument `(myevent, )` to the constructor of `EventQueueABM`. The constructor expects a tuple of events. The syntax `(x,)` is simply a quick way of making a 1-tuple.

```@example agents
dims = (10, 10)
space3 = GridSpace(dims)

model3 = EventQueueABM(myLearner,
                       (myevent, ), 
                       space3,
                       properties = Dict(:nd => distro1))

for i in 1:100
  add_agent_single!(model3,
                    LRPLearner(2, 0.01; W = [0.1, 0.9], A = [0.0 0.1; 0.2 0.0]), 
                    0,
                    get_lifetime(model3.nd))
end

adf3, mdf3 = run!(model3, 10_000_000; adata)

Plots.plot(adf3.time, adf3.mean_first_weight, label="mean W1")

savefig("meanweightExp.png") # hide
nothing # hide
```

![](meanweightExp.png)

The evolution of the mean may not look too different from what we observed in the (first) discrete-time simulation. However, observe that things are roughly one order of magnitude slower.

We can also illustrate the jump process nature of the continuous-time simulation by contrasting the two on short timescales:

```@example agents
# discrete-time simulation
Plots.plot(adf.time[1:100], adf.mean_first_weight[1:100], label="mean W1")

savefig("meanweight_short.png") # hide
nothing # hide
```

![](meanweight_short.png)

```@example agents
# continuous-time simulation
Plots.plot(adf3.time[1:100], adf3.mean_first_weight[1:100], label="mean W1")

savefig("meanweightExp_short.png") # hide
nothing # hide
```

![](meanweightExp_short.png)


