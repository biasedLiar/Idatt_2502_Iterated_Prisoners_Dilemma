def get_action(history, m):
    decision = 1
    if history.shape[1] >= 1 and history[1, -1] == 0:  # Choose to defect if and only if the opponent just defected.
        decision = 0
    return decision, None
