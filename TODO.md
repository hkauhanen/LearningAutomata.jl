- sanity checks for constructors
- sanity checks for interact! and simulate! (need to check agent's have the same dimensionalities and so on)
- implement an LRILearner (linear reward-inaction) - we get this basically for free from LRPLearner
- implement an LIPLearner (linear inaction-penalty)?
- implement mean dynamic utilities for LRPLearner
- proofread all LRPLearner code especially now that we've moved to the new Greek names for the arguments. Work through the mathematics again to make sure all the matrix algebra is correct
- implement package tests
- design and draw a logo
- proper handling of random numbers (learner and environment types need to be given a field that points to an RNG; routines such as act and punishes need to consult those RNGs)
- deprecate simulate! in favour of interact!
- missing docstrings


BUGS:

- costs do not work. troubleshoot and fix this
