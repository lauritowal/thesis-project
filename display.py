'''#https://stackoverflow.com/questions/60683901/tensorboard-smoothing

import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns

from scipy.interpolate import interp1d


sns.set()

dfs = {}
dfs[0] = pd.read_csv(r"/content/drive/MyDrive/thesis_graphs/sincos/run-TD3_guidance-continuous-v0_2021-06-26_08-34-403y3x4f42-tag-ray_tune_episode_reward_mean.csv")
dfs[1] = pd.read_csv(r"/content/drive/MyDrive/thesis_graphs/sincos/run-TD3_guidance-continuous-v0_2021-07-23_16-44-36wz9_dn8_-tag-ray_tune_episode_reward_mean.csv")
dfs[2] = pd.read_csv(r"/content/drive/MyDrive/thesis_graphs/sincos/run-TD3_guidance-continuous-v0_2021-07-24_22-54-2108w_5fiq-tag-ray_tune_episode_reward_mean.csv")


df_mean = pd.concat([dfs[0], dfs[1], dfs[2]]).groupby(level=0).mean()

# print("df_mean", df_mean)

#df_mean.plot(x=1, y=2)
df_mean.head()
# sns.lineplot(data=df_mean, x=1, y=2)

sns.lineplot(data=df_mean, x="Step", y="Value")



plt.show()


################################
print(df_mean)

ts_factor = 0.99 # 0.5, 0.85
smooth = df_mean.ewm(alpha=(1 - ts_factor)).mean()

g1 = sns.lineplot(data=df_mean, x="Step", y="Value", alpha=0.2, color="blue")
g2 = sns.lineplot(data=smooth, x="Step", y="Value", color="blue")

g1.set(xlim=(0, 0.97e7))
g2.set(xlim=(0, 0.97e7))

#plt.plot(df_mean["Value"], alpha=0.3, color="red")
#plt.plot(smooth["Value"], color="red")
plt.title("Tensorboard Smoothing = {}".format(TSBOARD_SMOOTHING[ptx]))
plt.grid(alpha=0.3)

name = "default"
plt.savefig(f'/content/drive/MyDrive/thesis_graphs/svgs/{name}.svg')

plt.show()
'''