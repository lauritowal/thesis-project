import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

df = pd.read_json (r'run-TD3_guidance-continuous-v0_2021-05-24_09-28-43x24t3rud-tag-ray_tune_custom_metrics_num_on_track_mean.json')
print(df)
df.plot(x=1, y=2)
df.head()
sns.lineplot(data=df, x=1, y=2)

plt.show()