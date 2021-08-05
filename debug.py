import pandas as pd 

df = pd.read_csv('processed_out_acc.csv')
df.columns = ['label1', 'label2', 'image1', 'image2', 'rt', 'acc']

rt_max = df['rt'].max()
rt_min = df['rt'].min()


print(rt_max)
print(rt_min)