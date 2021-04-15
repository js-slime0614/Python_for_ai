import numpy as np

def solution(numbers):
    answer = []
    list_set = list()
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j:
                list_set.append(numbers[i] + numbers[j])
    answer = set(list_set)
    
    return answer
numbers = [2,1,3,4,1]
print(solution(numbers))