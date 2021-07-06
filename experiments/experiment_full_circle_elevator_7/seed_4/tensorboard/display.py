import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

import seaborn as sns
sns.set()

df = pd.read_json (r'run-TD3_guidance-continuous-v0_2021-06-14_13-46-260fmowqi_-tag-ray_tune_evaluation_episode_reward_mean.json')
# df = pd.read_json (r'run-TD3_guidance-continuous-v0_2021-06-14_13-46-260fmowqi_-tag-ray_tune_custom_metrics_num_on_track_mean.json')
print(df)
df.plot(x=1, y=2)
df.head()
sns.lineplot(data=df, x=0, y=2, hue=1)

plt.show()



plt.show()