import numpy as np
arr = np.array([[1., 2, 3], [1, 2, 3]])
arr.dtype
print(arr.dtype)
#숫자가 하나라도 float 라면?
#float64 가 된다
#.astype 으로 바꿔줄수있음
arr = arr.astype(np.int32)
print(arr.dtype)

arr = np.array([[1., 2, 3], [1, 2, 3]], dtype=np.uint8)
print(arr.dtype)

#shape
print(arr.shape)

#차원확인하는방법
print(len(arr.shape))
print(arr.ndim)

#size도 확인가능
print(arr.size)

#reshape
#말그대로 재구성 차원도 바꿔줄수잇음