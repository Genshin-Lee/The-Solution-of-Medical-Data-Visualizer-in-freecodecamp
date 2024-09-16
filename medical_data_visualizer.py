import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(df["weight"] / (df["height"]/100)**2 > 25,1,0)

# 3
df["cholesterol"] = df["cholesterol"].apply(lambda x: 1 if x > 1 else 0)
df["gluc"] = df["gluc"].apply(lambda x: 1 if x > 1 else 0)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df,id_vars=['cardio'],value_vars= ["cholesterol", "gluc", "smoke", "alco", "active", "overweight"], var_name='Category', value_name='Value')

    # 6
    df_cat["Value_type"] = df_cat["Value"]
    df_cat = df_cat.groupby(['cardio',"Category","Value_type"]).count()
 
    # 7
    sns.catplot(x="Category", y="Value", hue="Value_type", kind="bar", col="cardio", data=df_cat)


    # 8
    fig = sns.catplot(x="Category", y="Value", hue="Value_type", kind="bar", col="cardio", data=df_cat)
    fig.set_axis_labels("variable","total")
    fig = fig.fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                (df['height'] >= df['height'].quantile(0.025)) &
                (df['height'] <= df['height'].quantile(0.975)) &
                (df['weight'] >= df['weight'].quantile(0.025)) &
                (df['weight'] <= df['weight'].quantile(0.975))
                ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones(corr.shape), k = 0).astype(bool)



    # 14
    fig, ax = plt.subplots(figsize = (10,10))

    # 15
    sns.heatmap(corr,mask = mask,annot=True,fmt = ".1f",ax =ax,center = 0,cbar=True)

    # 16
    fig.savefig('heatmap.png')
    return fig
