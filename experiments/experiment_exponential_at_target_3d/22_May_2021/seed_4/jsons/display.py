import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

df = pd.read_json (r'run-TD3_guidance-continuous-v0_2021-05-22_23-59-48alvvifat-tag-ray_tune_episode_reward_mean.json')
print(df)
df.plot(x=1, y=2)

plt.show()