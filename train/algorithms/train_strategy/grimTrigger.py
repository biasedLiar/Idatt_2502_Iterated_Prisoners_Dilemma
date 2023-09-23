# The grim trigger will start off always cooperating.
# However, once the opponent defects once,
# the grim trigger will defect for the rest of the game,
# no matter what the opponent does.

def get_action(history, mem):
    triggered = False
    if mem is not None and mem:
        triggered = True
    else:
        if history.shape[1] >= 1 and history[1, -1] == 0:
            triggered = True

    if triggered:
        return 0, True
    else:
        return 1, False
