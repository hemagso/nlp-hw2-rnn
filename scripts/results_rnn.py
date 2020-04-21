import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df_gradients = pd.read_csv("../models/rnn_01/gradients.csv", index_col=0)
df_loss = pd.read_csv("../models/rnn_01/gradients.csv", index_col=0)


fig, axs = plt.subplots(5, 1, sharex=True)
for idx, parameter in enumerate(df_gradients.columns):
    sns.distplot(df_gradients[parameter], ax=axs[idx])
# plt.show()

df_gradients.reset_index(inplace=True)
df_gradients["group"] = df_gradients["index"] // 50
df_gradients.set_index("index", inplace=True)

df_avg = df_gradients.groupby("group").mean().reset_index()
print(df_avg)

fig, axs = plt.subplots(5, 1, sharex=True)
for idx, parameter in enumerate(df_gradients.columns):
    if parameter != "group":
        sns.lineplot(x=df_avg["group"], y=df_avg[parameter], ax=axs[idx])
plt.show()