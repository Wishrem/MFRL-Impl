import numpy as np

def is_same_array(lst, target, require_order=False):
    lst = np.array(lst)
    target = np.array(target)
    
    if len(lst) != len(target):
        return False
    if not require_order:
        for i in range(len(lst)):
            flag = False
            for j in range(len(target)):
                if np.all(lst[i] == target[j]):
                    flag = True
                    break
            if not flag:
                return False
        return True
    
    return np.all(lst == target)