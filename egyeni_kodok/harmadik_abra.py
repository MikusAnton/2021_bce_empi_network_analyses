"""Visualization function examples for the homework project"""

def extract_feature_to_degree_centrality_standard_error(nodes):
    # This function outputs the results of a pivot-like-query. 
    # The output is 8 lists, the 1st is the age buckets, the second is the average degree centrality, the third ad fouurth the confodence intervals
    
    import pandas as pd
    import math
    import numpy as np

    summary = nodes.copy()

    summary["gender"] = summary["gender"].map({1.0:"Male", 0.0:"Female"})

    temp_data = pd.DataFrame()

    #For pivot-like calculation purposes
    temp_data["id_occurances"] =  edges.stack()
    temp_data.drop(columns = temp_data.columns[temp_data.columns!="id_occurances"], inplace = True)
    temp_data = temp_data.reset_index()
    temp_data = temp_data.groupby(by = "id_occurances", axis = 0, as_index = True).count().rename(columns={"level_0":"Number_of_acquintances"})
    temp_data.drop(columns = temp_data.columns[temp_data.columns!="Number_of_acquintances"], inplace = True)

    #Adding back the results into the dataframe
    summary = summary.join(other = temp_data, how = 'left', on = 'user_id')
    summary.head(10)

    #Groping the results according to age, and calculating mean - standard deviation, frequency and standard error as needed
    table_of_the_means = summary[ ["AGE", "gender", "Number_of_acquintances"] ].groupby(["AGE", "gender"]).mean().rename(columns={"Number_of_acquintances":"mean"})
    table_of_the_deviations = summary[ ["AGE", "gender", "Number_of_acquintances"] ].groupby(["AGE", "gender"]).std().rename(columns={"Number_of_acquintances":"std"})
    table_of_obs_numbers = summary[ ["AGE","gender", "Number_of_acquintances"] ].groupby(["AGE", "gender"]).count().rename(columns={"Number_of_acquintances":"count"})
    table_of_results = table_of_the_means.join(other = table_of_the_deviations, how = 'left')
    table_of_results = table_of_results.join(other = table_of_obs_numbers, how = 'left')
    table_of_results["standard_error"] = 1.96 * ( table_of_results["std"] / table_of_results["count"].apply(np.sqrt) )

    #Generating output
    ##male_lists for graph
    list_of_male_ages = list(table_of_results.xs("Male", level = 1).index)
    list_of_male_means = list(table_of_results.xs("Male", level = 1)["mean"])
    list_of_male_st_errors = list(table_of_results.xs("Male", level = 1)["standard_error"])
   
    ##female_lists for graph
    list_of_female_ages = list(table_of_results.xs("Female", level = 1).index)
    list_of_female_means = list(table_of_results.xs("Female", level = 1)["mean"])
    list_of_female_st_errors = list(table_of_results.xs("Female", level = 1)["standard_error"])
    
    return list_of_male_ages, list_of_male_means, list_of_male_st_errors, list_of_female_ages, list_of_female_means, list_of_female_st_errors

def get_feature_summmary_degree_centrality(G):
    import networkx as nx
    import pandas as pd

    possible_number_of_fokszam = len(G.nodes) - 1
    intermediate_data = pd.DataFrame({'degree_centralities':nx.degree_centrality(G)}) 
    intermediate_data['degree_centralities'] = intermediate_data['degree_centralities'] * possible_number_of_fokszam

    return intermediate_data, 'degree_centralities'

def get_feature_summmary_neighbor_connectivity(G):
    import networkx as nx
    import pandas as pd

    intermediate_data = pd.DataFrame({'neighbor_connectivity':nx.average_neighbor_degree(G)}) 
    intermediate_data['neighbor_connectivity'] = intermediate_data['neighbor_connectivity']

    return intermediate_data, 'neighbor_connectivity'

def get_feature_summmary_triadic_closure(G):
    import networkx as nx
    import pandas as pd

    intermediate_data = pd.DataFrame({'triadic_closure':nx.clustering(G)}) 
    intermediate_data['triadic_closure'] = intermediate_data['triadic_closure']

    return intermediate_data, 'triadic_closure'

def extract_feature_to_age_and_gender(nodes, feature_summary, feature_name):
    # This function outputs the results of a pivot-like-query. 
    # The output is 6 lists, the 1st is the age buckets, the second is the average degree centrality, the third is the 95%-conf-scaled symmetrical error
    
    import pandas as pd
    import math
    import numpy as np

    summary = nodes.copy()

    summary["gender"] = summary["gender"].map({1.0:"Male", 0.0:"Female"})

    #Adding back the results into the dataframe
    summary = summary.join(other = feature_summary, how = 'left', on = 'user_id')
    summary.head(10)

    #Groping the results according to age, and calculating mean - standard deviation, frequency and standard error as needed
    table_of_the_means = summary[ ["AGE", "gender", feature_name] ].groupby(["AGE", "gender"]).mean().rename(columns={feature_name:"mean"})
    table_of_the_deviations = summary[ ["AGE", "gender", feature_name] ].groupby(["AGE", "gender"]).std().rename(columns={feature_name:"std"})
    table_of_obs_numbers = summary[ ["AGE","gender", feature_name] ].groupby(["AGE", "gender"]).count().rename(columns={feature_name:"count"})
    table_of_results = table_of_the_means.join(other = table_of_the_deviations, how = 'left')
    table_of_results = table_of_results.join(other = table_of_obs_numbers, how = 'left')
    table_of_results["standard_error"] = 1.96 * ( table_of_results["std"] / table_of_results["count"].apply(np.sqrt) )

    #Generating output
    ##male_lists for graph
    list_of_male_ages = list(table_of_results.xs("Male", level = 1).index)
    list_of_male_means = list(table_of_results.xs("Male", level = 1)["mean"])
    list_of_male_st_errors = list(table_of_results.xs("Male", level = 1)["standard_error"])
   
    ##female_lists for graph
    list_of_female_ages = list(table_of_results.xs("Female", level = 1).index)
    list_of_female_means = list(table_of_results.xs("Female", level = 1)["mean"])
    list_of_female_st_errors = list(table_of_results.xs("Female", level = 1)["standard_error"])
    
    return list_of_male_ages, list_of_male_means, list_of_male_st_errors, list_of_female_ages, list_of_female_means, list_of_female_st_errors

def create_the_third_plot(G, nodes, edges):
    #creating plot
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.style.use('ggplot')

    fig, ax = plt.subplots(2 , 2, squeeze = False,  figsize=(12,6.5))

        #Bulding graph
    #1ST AXIS
        #Get data for graph
    support_variable_for_output = get_feature_summmary_degree_centrality(G)
    list_of_male_ages, list_of_male_means, list_of_male_st_errors, list_of_female_ages, list_of_female_means, list_of_female_st_errors = \
            extract_feature_to_age_and_gender(nodes, support_variable_for_output[0], support_variable_for_output[1])
        #Drawing
    ax[0,0].set_title('Degree_centralities')
    ax[0,0].errorbar(x = list_of_male_ages, y = list_of_male_means, yerr = list_of_male_st_errors, marker='o', 
                     mfc='blue', mec='blue', capthick = 1, ms=1, capsize = 3, c = 'blue', label = 'Male')
    ax[0,0].errorbar(x = list_of_female_ages, y = list_of_female_means, yerr = list_of_female_st_errors, marker='o', 
                     mfc='pink', mec='pink', capthick = 1, ms=1, capsize = 3, c = 'pink', label = 'Female')
    ax[0,0].legend(loc ="upper right")
    ax[0,0].set_xlabel('Age')
    ax[0,0].set_ylabel('Degree')

    #2ND AXIS
        #Get data for graph
    support_variable_for_output = get_feature_summmary_neighbor_connectivity(G)
    list_of_male_ages, list_of_male_means, list_of_male_st_errors, list_of_female_ages, list_of_female_means, list_of_female_st_errors = \
            extract_feature_to_age_and_gender(nodes, support_variable_for_output[0], support_variable_for_output[1])
        #Drawing
    ax[0,1].set_title('Neighbor Connectivity')
    ax[0,1].errorbar(x = list_of_male_ages, y = list_of_male_means, yerr = list_of_male_st_errors, marker='o', 
                     mfc='blue', mec='blue', capthick = 1, ms=1, capsize = 3, c = 'blue', label = 'Male')
    ax[0,1].errorbar(x = list_of_female_ages, y = list_of_female_means, yerr = list_of_female_st_errors, marker='o', 
                     mfc='pink', mec='pink', capthick = 1, ms=1, capsize = 3, c = 'pink', label = 'Female')
    ax[0,1].legend(loc ="upper right")
    ax[0,1].set_xlabel('Age')
    ax[0,1].set_ylabel('Neighbor Connectivity')

    #3RD AXIS
        #Get data for graph
    support_variable_for_output = get_feature_summmary_triadic_closure(G)
    list_of_male_ages, list_of_male_means, list_of_male_st_errors, list_of_female_ages, list_of_female_means, list_of_female_st_errors = \
            extract_feature_to_age_and_gender(nodes, support_variable_for_output[0], support_variable_for_output[1])
        #Drawing
    ax[1,0].set_title('Triadic Closure')
    ax[1,0].errorbar(x = list_of_male_ages, y = list_of_male_means, yerr = list_of_male_st_errors, marker='o', 
                     mfc='blue', mec='blue', capthick = 1, ms=1, capsize = 3, c = 'blue', label = 'Male')
    ax[1,0].errorbar(x = list_of_female_ages, y = list_of_female_means, yerr = list_of_female_st_errors, marker='o', 
                     mfc='pink', mec='pink', capthick = 1, ms=1, capsize = 3, c = 'pink', label = 'Female')
    ax[1,0].legend(loc ="upper right")
    ax[1,0].set_xlabel('Age')
    ax[1,0].set_ylabel('CC')

    #4TH AXIS
        #no embeddeddness function - no plot

    fig.subplots_adjust(hspace=1.0, wspace = 0.25)
    return fig.show()
