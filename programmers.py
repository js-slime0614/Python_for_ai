import numpy as np

def solution(numbers):
    answer = []
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j :
                answer.append(numbers[i] + numbers[j])
    for i in range(len(answer)):
        for j in range(len(answer)):
            if answer[i] == answer[j] and i != j:
                answer[j] = 0
    answer.sort()
    answer.remove(0)
    return answer
numbers = [2,1,3,4,1]
print(solution(numbers))
#test