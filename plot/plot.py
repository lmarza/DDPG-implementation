# check this https://www.youtube.com/watch?v=GcXcSZ0gQps&t=2096s for more details

import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

sns.set_style("darkgrid")
result = pd.read_csv('result.csv')
ax = sns.lineplot(x="Episode", y="Reward", data=result, ci= 'sd')
plt.title('DDPG implementation')
plt.xlabel('No of episodes')
plt.ylabel('Success rate (%)')
plt.legend(loc='lower right')
plt.show(ax)
