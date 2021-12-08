"""Functions to load and filter the network"""

def get_number_of_male_and_female_friends_of_users(nodes, edges):
    #Outputs two dictinoaries, with the key of user ID-s the dictinaries of nr of male friends and nr of female friends respectively
    import pandas as pd
    
    gender_dict = dict(zip(nodes["user_id"], nodes["gender"]))

    temp_data1 = edges.copy()
    temp_data2 = nodes.copy()
    temp_data2['AGE'] = temp_data2['AGE']
    temp_data2.drop(columns = ['public', 'region', 'TRAIN_TEST'], inplace = True)
            #Creating dictonaries for mapping
    temp_data2["gender"] = temp_data2["gender"].fillna(2.0)
    temp_data2["gender"] = temp_data2["gender"].map({1.0:"Male", 0.0:"Female", 2.0:"Unknown"})
    gender_dict = dict(zip(temp_data2["user_id"], temp_data2["gender"]))                                  
            #Execeuting the mapping into a dataframe ready to be query-ed
    temp_data1['smaller_gender'] = temp_data1["smaller_id"].map(gender_dict)
    temp_data1['greater_gender'] = temp_data1["greater_id"].map(gender_dict)
    
    summary_part1 = temp_data1.groupby(['smaller_id', 'smaller_gender', 'greater_gender']).count().reset_index(level=[0,1,2])
    summary_part2 = temp_data1.groupby(['greater_id', 'smaller_gender', 'greater_gender']).count().reset_index(level=[0,1,2])               
            #rename for merge
    summary_part1.rename(
            columns={'smaller_id':'user_id', 'smaller_gender':'user_gender', 'greater_gender':'friend_gender',
                    'greater_id':'friend_count'}, inplace = True)
    summary_part2.rename(
            columns={'greater_id':'user_id', 'greater_gender':'user_gender', 'smaller_gender':'friend_gender',
                    'smaller_id':'friend_count'}, inplace = True)

    summary_part1 = summary_part1.merge(summary_part2, 
                                            how = 'outer', 
                                            on = ['user_id', 'user_gender', 'friend_gender']).fillna(0)                                  
    summary_part1['friend_count'] = summary_part1['friend_count_x'] + summary_part1['friend_count_y']
    summary_part1.drop(columns = ['friend_count_x', 'friend_count_y'], inplace = True)
    summary_part1['friend_count'] = summary_part1['friend_count'].fillna(0)
    #summary_part1 = summary_part1.groupby(['user_gender', 'friend_gender']).sum().reset_index(level=[0,1])

    number_of_users_male_friends_dict = dict(zip(summary_part1.loc[summary_part1['friend_gender']=='Male',"user_id"], 
                                                 summary_part1.loc[summary_part1['friend_gender']=='Male',"friend_count"]))
    number_of_users_female_friends_dict = dict(zip(summary_part1.loc[summary_part1['friend_gender']=='Female',"user_id"], 
                                                 summary_part1.loc[summary_part1['friend_gender']=='Female',"friend_count"]))

    return number_of_users_male_friends_dict, number_of_users_female_friends_dict

def get_test_and_train_data_into_required_format(nodes, edges):
    #the required format means a 3-column matrix where the columns are the user gender, nr of male, and nr of female friends for train
    #For test, gender is obviously missing - since thatshould be n.a everywhere 
    import pandas as pd
    number_of_users_male_friends_dict, number_of_users_female_friends_dict = get_number_of_male_and_female_friends_of_users(nodes, edges)

    train_data = nodes[ nodes['TRAIN_TEST'] == 'TRAIN' ]
    test_data = nodes[ nodes['TRAIN_TEST'] == 'TEST' ]
        #train data manipulation
    train_data['Male_friend_count'] = train_data['user_id'].map(number_of_users_male_friends_dict, na_action=0)
    train_data['Female_friend_count'] = train_data['user_id'].map(number_of_users_female_friends_dict, na_action=0)
    train_data['Male_friend_count'] =  train_data['Male_friend_count'].fillna(0)
    train_data['Female_friend_count'] = train_data['Female_friend_count'].fillna(0)
    
    train_data.drop(columns = ['user_id', 'public', 'region', 'AGE', 'TRAIN_TEST'], inplace = True)
    
        #test data manipulation
    test_data['Male_friend_count'] = test_data['user_id'].map(number_of_users_male_friends_dict, na_action=0)
    test_data['Female_friend_count'] = test_data['user_id'].map(number_of_users_female_friends_dict, na_action=0)
    test_data['Male_friend_count'] = test_data['Male_friend_count'].fillna(0)
    test_data['Female_friend_count'] = test_data['Female_friend_count'].fillna(0)
    
    test_data.drop(columns = ['public', 'region', 'AGE', 'TRAIN_TEST', 'gender'], inplace = True)
    
    return train_data, test_data

def decision_tree_classification(nodes, edges):
    #Based on a a train data, makes a forecast on a test data. All model building happens inside, the classification method is
    #a plain vanilla decision tree
    
    #The output is a dictionary, which matches the user ID-s as keys, with the predicted gender, as values 
    
    # !!!! A ML-KÓD NEM SAJÁT, INNEN COPY-PASTA: https://www.w3schools.com/python/python_ml_decision_tree.asp!!!!
    
    import pandas as pd
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    import matplotlib.pyplot as plt
    import matplotlib.image as pltimg
    
        #Datasets 
    print("Preparing data...")
    train_data, test_data = get_test_and_train_data_into_required_format(nodes, edges)
    
        #Modell building
    print("Data prepared, building the model...")
    X = train_data[['Male_friend_count', 'Female_friend_count']]
    y = train_data['gender']
    
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(X, y)
    data = tree.export_graphviz(dtree, out_file=None, feature_names=['Male_friend_count', 'Female_friend_count'])
    
        #Some communication with the user
    #print('The calssifying decision tree followed the following criteria:')
    #img=pltimg.imread('mydecisiontree.png')
    #imgplot = plt.imshow(img)
    #plt.show()
    
    print('Making the predictions...')
    test_data['gender'] = dtree.predict(test_data[['Male_friend_count', 'Female_friend_count']])
    
    print(f"The model predicted {test_data['gender'].sum()} males, and {test_data['gender'].count() - test_data['gender'].sum()} females")
    
    return dict(zip(test_data['user_id'], test_data['gender']))
