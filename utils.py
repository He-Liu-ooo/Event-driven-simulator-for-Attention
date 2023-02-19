""" meta time of the whole system, in nanoseconds """
METATIME = 0.1 


""" 
States of the data in the core SRAM 

READY: data that ready to engage in calculation
REMOVE: data that wait to be removed into GB
REMOVING: data that are removing into GB

READY -> REMOVE -> REMOVING -> READY
"""
READY = 0
REMOVE = 1  # NULL
REMOVING = 2


"""  
States of the data in the core array

NULL: no data
SUBSUM: data in the array that is only subsum, not ready for removing yet
REMOVING: data that are removing into GB
COMPLETE: data in the array that is complete sum, should be removed as soon as possible

NULL -> SUBSUM -> COMPLETESUM -> REMOVING -> NULL
"""
NULL = 0
SUBSUM = 1
# REMOVING = 2
COMPLETESUM = 3


""" 
States of the data of A/X in the core for Q*K/LP

NULL: no data(data of A/X is not calculated yet)
A/X: data of A/X is calculated, can be used for executing softmax/layernorm
A_CAL/X_CAL: data of A/X in GB that is being calculated by softmax/layernorm unit
A_SOFTMAX/X_LAYERNORM: data undergoes softmax/layernorm, can be used for A'*V/FC1 calculation

GB(Softmax): NULL -> A -> REMOVING -> A_CAL -> REMOVING -> A_SOFTMAX
GB(Layernorm): NULL -> X -> REMOVING -> X_CAL
Softmax: NULL -> A -> A_SOFTMAX -> NULL
Layernorm: NULL -> X -> X_LAYERNROM -> REMOVING -> NULL
"""
# NULL = 0
A = 1  # X
# REMOVING = 2
A_CAL = 3   # X_CAL
A_SOFTMAX = 4   #X_LAYERNORM
