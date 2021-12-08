import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def piecewise_list_multiplication(list_to_modify, multiplicator):
    import numpy as np
    for indexer, element in zip( np.arange(len(list_to_modify)), list_to_modify):
        list_to_modify[indexer] = element * multiplicator
        
    return list_to_modify

def bucketing_pivot_table(pivot_like_data, gender):
    #male/female bucketing, returns the data in the required format
    import pandas as pd
    
    output = pd.DataFrame()
    ages = pivot_like_data.index[ pivot_like_data.index > 0]
    age_negation_if_female = 1
    
    bucket_1 = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    bucket_1_name_add_on = "(-5Yrs)<= X <= (+5Yrs)"
    bucket_2 = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    bucket_2_name_add_on = "(+20Yrs)<= X <= (+30Yrs)"
    bucket_3 = [-30, -29, -28, -27, -26, -25, -24, -23, -22, -21, -20]
    bucket_3_name_add_on = "(-20Yrs)<= X <= (-30Yrs)"
    
    if gender == 'Female':
                opposed_gender = 'Male'
    if gender == 'Male':
                opposed_gender = 'Female'

            
        # Aggregations start here
    for buckets, bucket_name_addon in zip([bucket_1, bucket_2, bucket_3], [bucket_1_name_add_on, bucket_2_name_add_on, bucket_3_name_add_on]):
        
        for negate_gender in [1, -1]:
            if negate_gender == -1:
                gender_for_bucket = opposed_gender
            else:
                gender_for_bucket = gender

            
            offseting_values = piecewise_list_multiplication(buckets, negate_gender)
                
            bucket_name = f"{gender_for_bucket}: {bucket_name_addon}"
            
            for ages in pivot_like_data.columns:
                sum_in_bucket = 0

                for offset in buckets:
                    if (ages * negate_gender + offset) in pivot_like_data.index:
                        sum_in_bucket = sum_in_bucket + max(pivot_like_data.loc[ages * negate_gender + offset, ages] * negate_gender, 0)
                
                output.loc[ages, bucket_name] = sum_in_bucket

        #Normalization
    total_friend_at_age_grp_dictionary = dict(pivot_like_data.apply(abs).sum())
    output = output.fillna(0)
    
    for i in output.index:
        for j in output.columns:
            output.loc[i, j] = output.loc[i, j] / total_friend_at_age_grp_dictionary[i]
                
    return output
    
    
