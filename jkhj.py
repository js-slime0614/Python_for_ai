import pandas as pd

#make csv file in DeskTop
data = [[1,2,3,4],[5,6,7,8]]
df = pd.DataFrame(data)
df.to_csv('C:\\Users\\user\\Desktop\\ddd.csv')