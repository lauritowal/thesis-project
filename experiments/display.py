import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

df = pd.read_json (r'./experiment_full_circle_elevator_7/seed_4/tensorboard/TD3_guidance-continuous-v0_2021-06-14_13-46-260fmowqi_')
print(df)
df.plot(x=1, y=2)

plt.show()