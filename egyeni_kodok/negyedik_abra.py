import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def symmetrical_colormap(cmap_settings, new_name = None ):
    ''' This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold.
    '''
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_"+cmap_settings[0]  # ex: 'sym_Blues'
    
    # this defined the roughness of the colormap, 128 fine
    n= 128 
    
    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, n))    # take the standard colormap # 'right-part'
    colors_l = colors_r[::-1]                # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    colors = np.vstack((colors_l, colors_r))
    #colors = np.concatenate(colors_l[0:n/2],colors_l[n/2:n]) 
    mymap = mcolors.LinearSegmentedColormap.from_list(new_name, colors)

    return mymap

def plot_male_connections(m_df):

    import matplotlib.pyplot as plt

    #get and shape data
    male_data = m_df.pivot("friend_age", "user_age", "friend_score")

    #normalization
    male_data = male_data / male_data.sum().sum()

    x_ticks_male_data = list(np.arange(0, len(male_data.columns)))
    y_ticks_male_data = list(np.arange(0, len(male_data.index)))

    fig, ax = plt.subplots(1,1, figsize = (20,15))
    fig.suptitle('Total of connections for males at the given age-pair. Negative ages denote opposite sex')

    cmap_settings = ('RdPu', None)
    my_colormap = symmetrical_colormap(cmap_settings= cmap_settings, new_name =None)

        #Graphing males
    pcm1 = ax.pcolor(male_data, cmap = my_colormap)
    ax.set_xlabel('Age (Male)')
    ax.set_ylabel('Age of friends')
    ax.set_yticks(y_ticks_male_data  )
    ax.set_xticks(x_ticks_male_data   )

    ax.set_yticklabels(list(male_data.index))
    ax.set_xticklabels(list(male_data.columns) )

    fig.colorbar(pcm1)
    fig.subplots_adjust(wspace = 0.25)
    return fig.show()

def plot_female_connections(f_df):

    import matplotlib.pyplot as plt

    #get and shape data
    female_data = f_df.pivot("friend_age", "user_age", "friend_score")

    #normalization
    female_data = female_data / female_data.sum().sum()
    
    x_ticks_female_data = list(np.arange(0, len(female_data.columns)))
    y_ticks_female_data = list(np.arange(0, len(female_data.index)))


    fig, ax = plt.subplots(1,1, figsize = (20,15))
    fig.suptitle('Total of connections for females at the given age-pair. Negative ages denote opposite sex')

    cmap_settings = ('RdPu', None)
    my_colormap = symmetrical_colormap(cmap_settings= cmap_settings, new_name =None)

        #Graphing males
    pcm1 = ax.pcolor(female_data, cmap = my_colormap)
    ax.set_xlabel('Age (Female)')
    ax.set_ylabel('Age of friends')
    ax.set_yticks(y_ticks_female_data  )
    ax.set_xticks(x_ticks_female_data   )

    ax.set_yticklabels(list(female_data.index))
    ax.set_xticklabels(list(female_data.columns) )

    fig.colorbar(pcm1)
    fig.subplots_adjust(wspace = 0.25)
    return fig.show()

def give_me_the_dictionary_of_friends_based_on_ids(edges, nodes):
    gender_dict = {}
    temp_data1 = edges.copy()
    temp_data2 = nodes.copy()
    temp_data2['AGE'] = temp_data2['AGE']
    temp_data2.dropna()
    temp_data2.drop(columns = ['public', 'region', 'TRAIN_TEST'], inplace = True)
        #Creating dictonaries for mapping
    temp_data2["gender"] = temp_data2["gender"].map({1.0:"Male", 0.0:"Female"})
    age_dict = dict(zip(temp_data2["user_id"], temp_data2["AGE"]))  
    gender_dict = dict(zip(temp_data2["user_id"], temp_data2["gender"]))                                  
        #Execeuting the mapping into a dataframe ready to be query-ed
    temp_data1['smaller_gender'] = temp_data1["smaller_id"].map(gender_dict)
    temp_data1['greater_gender'] = temp_data1["greater_id"].map(gender_dict)
    temp_data1['smaller_age'] = temp_data1["smaller_id"].map(age_dict)
    temp_data1['greater_age'] = temp_data1["greater_id"].map(age_dict)
    
        #first gropby and results
    summary_part1 = temp_data1.groupby(['smaller_id', 'smaller_gender','smaller_age', 'greater_gender', 'greater_age']).count().reset_index(level=[0,1,2,3,4])
    summary_part2 = temp_data1.groupby(['greater_id', 'smaller_gender','smaller_age', 'greater_gender', 'greater_age']).count().reset_index(level=[0,1,2,3,4])               
        #rename for merge
    summary_part1.rename(
        columns={'smaller_id':'user_id', 'smaller_gender':'user_gender', 'smaller_age':'user_age', 'greater_gender':'friend_gender',
                'greater_age':'friend_age', 'greater_id':'friend_count'}, inplace = True)
    summary_part2.rename(
        columns={'greater_id':'user_id', 'greater_gender':'user_gender', 'greater_age':'user_age', 'smaller_gender':'friend_gender',
                'smaller_age':'friend_age', 'smaller_id':'friend_count'}, inplace = True)
    
    summary_part1 = summary_part1.merge(summary_part2, 
                                        how = 'outer', 
                                        on = ['user_id', 'user_gender', 'user_age', 'friend_gender', 'friend_age']).fillna(0)                                  
    summary_part1['friend_count'] = summary_part1['friend_count_x'] + summary_part1['friend_count_y']
    summary_part1.drop(columns = ['user_id', 'friend_count_x', 'friend_count_y'], inplace = True)
    summary_part1 = summary_part1.groupby(['user_gender', 'user_age', 'friend_gender', 'friend_age']).sum().reset_index(level=[0,1,2,3])

    #negating opposite gender
    dictonary_for_gender_negation = {'MaleMale':1, 'FemaleFemale':1, 'FemaleMale':-1, 'MaleFemale':-1}
    summary_part1['help_for_mapping'] = summary_part1['user_gender'] + summary_part1['friend_gender']
    summary_part1['help_for_mapping'] = summary_part1['help_for_mapping'].map(dictonary_for_gender_negation)
    summary_part1['friend_score'] = summary_part1['friend_count'] * summary_part1['help_for_mapping']
    summary_part1['friend_age'] = summary_part1['friend_age'] * summary_part1['help_for_mapping']
    summary_part1.drop(columns = ['help_for_mapping', 'friend_count', 'friend_gender'], inplace = True)
    
    #finalising
    male_summary = summary_part1[ summary_part1['user_gender'] =='Male' ].drop(columns = 'user_gender')
    female_summary = summary_part1[ summary_part1['user_gender'] =='Female' ].drop(columns = 'user_gender')
    
    color_scale_bottpm_for_males = male_summary['friend_score'].apply(abs)
    
    return male_summary, female_summary                                  

