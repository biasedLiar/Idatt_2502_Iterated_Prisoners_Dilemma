

def get_action(history, memory):
    decision = None
    if history.shape[1] == 0:
        decision = 1
    else:
        decision = history[0,
                           -1]
        if history[1,
                   -1] == 0:
            decision = 1-decision
            
    return decision, None