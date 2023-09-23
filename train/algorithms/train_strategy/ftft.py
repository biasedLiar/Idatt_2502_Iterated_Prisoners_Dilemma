# Forgiving tit-for-tat
# It will always cooperate unless the opponent's last two moves where defect.
def get_action(history, memory):
    choice = 1
    if history.shape[1] >= 2 and history[1, -1] == 0 and history[
        1, -2] == 0:  # We check the TWO most recent turns to see if BOTH were defections, and only then do we defect too.
        choice = 0
    return choice, None
