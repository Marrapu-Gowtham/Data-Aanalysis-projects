#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


deliveries_data = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/IPL Ball-by-Ball 2008-2020.csv')
match_data = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/IPL Matches 2008-2020.csv')


# In[4]:


match_data .head(4)


# In[5]:


match_data.columns


# In[6]:


match_data.isnull().sum()


# In[7]:


match_data.shape[0]


# In[8]:


type(match_data.shape)


# In[9]:


match_data['venue'].unique()


# In[10]:


match_data['city'].unique()


# In[11]:


match_data['team1'].unique()


# In[12]:


match_data['team2'].unique()


# In[13]:


match_data['toss_winner'].value_counts()


# In[14]:


match_data['toss_winner'].value_counts().index[0]


# In[15]:


match_data['player_of_match'].value_counts().index[0]


# In[ ]:





# In[16]:


deliveries_data.columns


# In[17]:


deliveries_data['batsman_runs'].unique()


# In[18]:


deliveries_data['batsman'].unique()


# In[19]:


df_kholi = deliveries_data[deliveries_data['batsman'] == 'V Kohli']


# In[20]:


df_kholi.columns


# In[21]:


df_kholi['dismissal_kind'].value_counts()


# In[22]:


df_kholi['batsman_runs'].unique()


# In[23]:


len(df_kholi[df_kholi['batsman_runs'] == 1])


# In[24]:


len(df_kholi[df_kholi['batsman_runs'] == 2]) * 2


# In[25]:


len(df_kholi[df_kholi['batsman_runs'] == 3]) * 3


# In[26]:


len(df_kholi[df_kholi['batsman_runs'] == 4]) * 4


# In[27]:


len(df_kholi[df_kholi['batsman_runs'] == 6]) * 6


# In[ ]:





# In[28]:


import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot , iplot


# In[29]:


values = [1919, 692, 39, 2016, 1212]
labels = [1, 2, 3, 4, 6]

trace = go.Pie(labels = labels, values = values, hole=0.3)
data = [trace]
fig = go.Figure(data = data)


# In[30]:


fig.show()


# In[ ]:





# In[31]:


match_data.columns


# In[32]:


match_data['date']


# In[33]:


match_data['Season'] = pd.to_datetime(match_data['date']).dt.year


# In[34]:


match_data.columns


# In[35]:


season_toss_count_df = match_data.groupby(['Season', 'toss_decision']).size().reset_index().rename(columns={0 : 'count'})


# In[36]:



#season_toss_count_df.plot(kind='bar', figsize=(8,6))


# In[37]:


plt.figure(figsize=(10,6))
sns.barplot(x ='Season', y ='count', hue= 'toss_decision', data = season_toss_count_df )


# In[ ]:





# In[38]:


match_data.columns


# In[40]:


match_data[['team1','team2', 'toss_winner', 'winner']]


# In[43]:


match_data['toss_win_game_win']= np.where(match_data['toss_winner']==match_data['winner'], 'yes', 'no')


# In[45]:


match_data['toss_win_game_win'].value_counts()


# In[46]:


match_data['toss_win_game_win'].value_counts().index


# In[47]:


match_data['toss_win_game_win'].value_counts().values


# In[59]:


values = match_data['toss_win_game_win'].value_counts().values
labels = match_data['toss_win_game_win'].value_counts().index
trace = go.Pie(labels = labels, values = values, hole=0.3)
data = [trace]
fig = go.Figure(data = data)
fig.update_traces(hoverinfo='label+percent', textinfo='label+percent')


# In[53]:


fig.show()


# In[60]:


match_data.columns


# In[62]:


match_data['Season'].unique()


# In[65]:


df_2018 = match_data[match_data['Season']==2018]


# In[74]:


df_2018['winner'].tail(1).values[0]


# In[ ]:





# In[82]:


winner_team = {}
for year in sorted(match_data['Season'].unique()):
    current_year_df = match_data[match_data['Season']==year]
    winner_team[year] = current_year_df ['winner'].tail(1).values[0]


# In[83]:


winner_team


# In[85]:


winner_team.values()


# In[90]:


from collections import Counter


# In[95]:


Counter(winner_team.values())


# In[ ]:





# In[96]:


match_data.columns


# In[97]:


match_data[['team1','team2']]


# In[105]:


matches_played = match_data['team1'].value_counts() + match_data['team2'].value_counts()


# In[107]:


matches_played


# In[108]:


type(matches_played)


# In[110]:


matches_played_df = matches_played.to_frame().reset_index()


# In[111]:


matches_played_df.columns = ['team_name', 'matches_played']


# In[112]:


matches_played_df


# In[114]:


wins = pd.DataFrame(match_data['winner'].value_counts()).reset_index()


# In[116]:


wins.columns = ['team_name', 'wins']


# In[117]:


wins


# In[148]:


played = matches_played_df.merge(wins, on = 'team_name', how='inner')


# In[149]:


played


# In[150]:


import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot , iplot


# In[151]:


'''trace1 = go.Bar(
    x = played['team_name'],
    y = played['matches_played'],
    name = 'Total Matches'  
)

trace2 = go.Bar(
    x = played['team_name'],
    y = played['wins'],
    name = 'Matches Won'    
)
data = [trace1 , trace2]
iplot(data)
'''


# In[167]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual DataFrame values)
# Example: played = pd.DataFrame({'team_name': ['Team A', 'Team B'], 'matches_played': [10, 15], 'wins': [6, 10]})
teams = played['team_name']
matches_played = played['matches_played']
wins = played['wins']

# Set up bar width and positions
bar_width = 0.38
x = np.arange(len(teams))

plt.figure(figsize=(20,12))

# Create the bars
plt.bar(x, matches_played, width=bar_width, label='Total Matches', color='b', align='center')
plt.bar(x + bar_width, wins, width=bar_width, label='Matches Won', color='g', align='center')

# Add labels and title
plt.xlabel('Team Name')
plt.ylabel('Number of Matches')
plt.title('Matches Played vs Matches Won')
plt.xticks(x + bar_width / 2, teams, rotation=45, ha='right')  # Center the x labels
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# In[168]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace this with your actual DataFrame values)
teams = played['team_name']
matches_played = played['matches_played']
wins = played['wins']

# Set up bar width and positions
bar_width = 0.35
x = np.arange(len(teams))
plt.figure(figsize=(20,12))
# Create the bars
plt.bar(x, matches_played, width=bar_width, label='Total Matches', color='b', align='center')
plt.bar(x + bar_width, wins, width=bar_width, label='Matches Won', color='g', align='center')

# Add labels and title
plt.xlabel('Team Name')
plt.ylabel('Number of Matches')
plt.title('Matches Played vs Matches Won')

# Adjust the x-ticks to add spacing
plt.xticks(x + bar_width / 2, teams, rotation=45, ha='right')  # Rotate and align labels

# Add some padding for better visibility
plt.subplots_adjust(bottom=0.2)

plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




