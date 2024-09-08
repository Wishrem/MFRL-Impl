import numpy as np

def is_same_array(lst, target, require_order=False):
    lst = np.array(lst)
    target = np.array(target)
    
    if len(lst) != len(target):
        return False
    if not require_order:
        for item in lst:
            for t_item in target:
                if np.all(item == t_item):
                    break
            else:
                return False
        return True
    return np.all(lst == target)