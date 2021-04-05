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
#size는 유지하되 구성을 바꿔줄수있다.
arr = np.array([[1, 2, 3], [1, 2, 3]])
arr = arr.reshape([6])
print(arr.shape)

#random array 생성
arr = np.random.randn(8,8)  
print(arr.shape)
arr = arr.reshape([32, 2])
print(arr.shape)

#arr.ravel arr 의 차원을 1차원으로 바꿔줌
#쫙 펼쳐서 1자로 나열해준다고 생각하면됨
arr = arr.ravel()
print(arr)

#size 값을 모르고 안의 값을 유지하되 차원을 늘리고 싶을땐?
#np.expand_dims(늘릴대상, 앞으로(0)or 뒤로(-1))
arr = np.expand_dims(arr, -1)

#np.zeros 0으로찬 배열 만들기
#자매품 = ones
#그럼 다른 수들은어케해?! 
#ones * 5 하면 5로이루어진배열이 된다
#np.arange(?) 0 부터 ?-1 만큼의 배열이 생성됨 (5, 9) 는 5부터 8까지