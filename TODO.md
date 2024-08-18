- add a 'reconfigure(x::LRPLearner, y::Dict)' function which will reset the
  parameters for x listed in y
- add an 'age' internal field for Learners which records number of times either
  reward or punish has been called
