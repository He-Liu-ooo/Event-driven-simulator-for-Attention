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
States of the data of A in the core for Q*K

NULL: no data(data of A is not calculated yet)
A: data of A is calculated, can be used for executing softmax
A_CAL: data of A in GB that is being calculated by softmax unit
A_SOFTMAX: data undergoes softmax, can be used for A'*V calculation

GB: NULL -> A -> REMOVING -> A_CAL -> REMOVING -> A_SOFTMAX
Softmax: NULL -> A -> A_SOFTMAX -> NULL
"""
# NULL = 0
A = 1
# REMOVING = 2
A_CAL = 3 
A_SOFTMAX = 4


""" 
State of the data of X in LN

NULL: no data
X: data of X has transferred to LN, waiting for calculation
X_CAL: data of X finishes calculation
REMOVING: data of calculated X is removing for next FC calculation 
"""
# NULL = 0
X = 1
X_CAL = 3
# REMOVING = 2