import pandas as pd
import numpy as np
import configargparse
from pathlib import Path
from utils import *

'''
mixed effects modeling approach (and some decisions used herein) derived from co-voting work by Ringe et al (2013): Ringe, Nils, Jennifer Nicoll Victor, and Justin H. Gross. "Keeping your friends close and your enemies closer? Information networks in legislative politics." British Journal of Political Science 43, no. 3 (2013): 601-628. 

'''

def load_feature_values_dicts(data_path):
    '''
    Reads the files containing actor-level feature values (scalars or vectors) and stores them in a dictionary - with the filename 
    providing the feature-name as the key. 
    
    Input:
    
    data_path: path to the data directory containing decisions.csv and subdirectory called features containing the various .csv files corresponding to the actor-level features
    
    
    Output:
    
    scalar_features_to_values: key: feature name (derived from csv file name); value: list of scalars (any datatype) quantifying that feature's value corresponding to every actor (in the same order as in decisions.csv)
    
    vector_features_to_values: key: feature name (derived from csv file name); value: 2-d matrix (float) quantifying that feature's value, rows corresponding to every actor (in the same order as in decisions.csv) and each actor in this case is quantified by a feature vector 
    '''
    
    features_base_path = Path(data_path) / 'features/'
    feature_files = Path(features_base_path).glob('*')
    scalar_features_to_values = {}
    vector_features_to_values = {}
    for f in feature_files:
        if '.csv' in str(f):
            df = pd.read_csv(f, header = None)
            if len(df.columns) == 1:
                scalar_features_to_values[f.stem] = list(df[0])
            else:
                vector_features_to_values[f.stem] = df.to_numpy()
    return scalar_features_to_values, vector_features_to_values

def get_codecision_agreement_rate(decisions1, 
                                  decisions2, 
                                  num_decisions):
    '''
    Get the codecision agreement rate given two lists of decisions (corresponding to actors 1 and 2): number of same decisions/number of total decisions
    
    Input: 
    decisions1: list of categorical decisions by actor 1
    decisions2: list of categorical decisions by actor 2 (always on corresponding items)
    num_decisions: total number of items on which each actor made some decision
    
    Output:
    agreement_rate (float): co-decision agreement rate
    '''
    
    num_same_decisions = 0
    for d1, d2 in zip(decisions1, decisions2):
        if d1 == d2:
            num_same_decisions += 1
    agreement_rate = num_same_decisions/num_decisions
    return agreement_rate

def get_scalar_pairwise_identity_val(val1, 
                                     val2):
    '''
    For scalar feature values corresponding to actor1 (val1) and actor2 (val2), return 1.0 if the values are equal, and -1.0 otherwise
    '''
    
    if val1 == val2:
        return 1.0
    else:
        return -1.0
    
def get_vector_pairwise_similarity_val(vec1, 
                                       vec2, 
                                       similarity_metric, 
                                       add_epsilon):
    '''
    For a pair of real-valued feature vectors, return the similarity value
    
    Input: 
    vec1: (float array) feature vector corresponding to actor1
    vec2: (float array) feature vector corresponding to actor2
    similarity_metric: (str) similarity metric to compute a score from pair of vector features for actors - currently, Cosine similarity is both the default and the only metric supported; implementation in utils.py
    add_epsilon: (boolean) whether to add a small epsilon value or not to all elements for a feature vector - helpful in case feature vectors can be all zeroes for any individual actor (which can result in null values when computing pairwise similarity)
    
    Output:
    similarity value (float)
    '''
    
    epsilon = 1e-7
    if add_epsilon:
        vec1 = vec1 + epsilon
        vec2 = vec2 + epsilon
    if similarity_metric == 'Cosine':
        return cosine_sim(vec1, vec2) #function imported from utils.py
    
def get_pairwise_data_elements(actor_ids, 
                               decision_vals, 
                               scalar_features_to_values, 
                               vector_features_to_values, 
                               similarity_metric, 
                               add_epsilon):
    '''
    Compute and store the various pairwise data elements - codecision agreement rate (y) and pairwise features (Xs) used to run the mixed effects model to estimate coefficients (fixed effects) of various features (X), to see how various features explain codecision agreement rate
    
    Input: 
    decision_vals: 2d matrix, with rows corresponding to individual actors, storing each actor's list of decisions
    
    scalar_features_to_values: key: feature name (derived from csv file name); value: list of scalars (any datatype) quantifying that feature's value corresponding to every actor (in the same order as in decisions.csv)
    
    vector_features_to_values: key: feature name (derived from csv file name); value: 2-d matrix (float) quantifying that feature's value, rows corresponding to every actor (in the same order as in decisions.csv) and each actor in this case is quantified by a feature vector 
    
    similarity_metric: (str) similarity metric to compute a score from pair of vector features for actors - currently, Cosine similarity is both the default and the only metric supported; implementation in utils.py
    
    add_epsilon: (boolean) whether to add a small epsilon value or not to all elements for a feature vector - helpful in case feature vectors can be all zeroes for any individual actor (which can result in null values when computing pairwise similarity)
    
    --- 
    
    Output:
    pairwise_actor_ids: IDs for the pairs of actors
    pairwise_codecision_agreement_rate: CoDecision Agreement rate for pairs of actors
    pairwise_scalar_feature_to_same_identity_vals: dict of scalar feature name to pairwise values for the feature value being same or not (1/-1) for pairs of actors
    pairwise_vector_feature_to_similarity_vals: dict of vector feature name to pairwise similarity values for pairs of actors
    removed_pairs: pairs who agreed or disagreed all the time are removed (and stored for documentation) following Ringe et al (2013)
    indiv_actors1: corr. list of actor 1 in the pair
    indiv_actors2: corr. list of actor 2 in the pair
    '''
    
    num_actors = decision_vals.shape[0]
    num_decisions = decision_vals.shape[1]
    
    #following Ringer et al. (2013) - we remove pairs who always voted together or never voted together.
    pairwise_actor_ids = []

    pairwise_codecision_agreement_rate = []

    pairwise_scalar_feature_to_same_identity_vals = {}
    pairwise_vector_feature_to_similarity_vals = {}
    for scalar_feat in scalar_features_to_values:
        pairwise_scalar_feature_to_same_identity_vals[scalar_feat] = []
    for vector_feat in vector_features_to_values:
        pairwise_vector_feature_to_similarity_vals[vector_feat] = []

    removed_pairs = []

    indiv_actors1, indiv_actors2 = [], []

    for i in range(num_actors):
        decisions1 = list(decision_vals[i])
        for j in range(i+1, num_actors):
            decisions2 = list(decision_vals[j])
            agreement_rate = get_codecision_agreement_rate(decisions1, decisions2, num_decisions)
            if agreement_rate > 0.0 and agreement_rate < 1.0:
                pairwise_codecision_agreement_rate.append(agreement_rate)
                for scalar_feat in scalar_features_to_values:
                    pairwise_scalar_feature_to_same_identity_vals[scalar_feat].append(get_scalar_pairwise_identity_val(scalar_features_to_values[scalar_feat][i],
                                                                                                                       scalar_features_to_values[scalar_feat][j]))

                for vector_feat in vector_features_to_values:
                    pairwise_vector_feature_to_similarity_vals[vector_feat].append(get_vector_pairwise_similarity_val(vector_features_to_values[vector_feat][i],
                                                                                                                      vector_features_to_values[vector_feat][j],
                                                                                                                      similarity_metric,
                                                                                                                      add_epsilon))

                pairwise_actor_ids.append(actor_ids[i] + '_' + actor_ids[j])
                indiv_actors1.append(actor_ids[i])
                indiv_actors2.append(actor_ids[j])
            else:
                removed_pairs.append((actor_ids[i], actor_ids[j]))
    assert len(pairwise_actor_ids) == len(pairwise_codecision_agreement_rate)
    
    return pairwise_actor_ids, pairwise_codecision_agreement_rate, pairwise_scalar_feature_to_same_identity_vals, pairwise_vector_feature_to_similarity_vals, removed_pairs, indiv_actors1, indiv_actors2

def get_pairwise_dataframe(pairwise_actor_ids, 
                           pairwise_codecision_agreement_rate, 
                           pairwise_scalar_feature_to_same_identity_vals, 
                           pairwise_vector_feature_to_similarity_vals, 
                           indiv_actors1, 
                           indiv_actors2,
                           similarity_metric):
    '''
    Using the output of the previous function, create a dataframe to be saved as a CSV in order to use it as input for R script fitting the mixed effects model: returns the dataframe storing the required pairwise columns.
    '''
    
    pairwise_data_df = pd.DataFrame()
    pairwise_data_df['Pair_ID'] = pairwise_actor_ids
    pairwise_data_df['Actor_1'] = indiv_actors1
    pairwise_data_df['Actor_2'] = indiv_actors2
    pairwise_data_df['CoDecision_Agreement_Rate'] = pairwise_codecision_agreement_rate
    for scalar_feat in pairwise_scalar_feature_to_same_identity_vals:
        pairwise_data_df['Same_' + scalar_feat] = pairwise_scalar_feature_to_same_identity_vals[scalar_feat]
    for vector_feat in pairwise_vector_feature_to_similarity_vals:
        pairwise_data_df[vector_feat + '_' + similarity_metric + '_Sim'] = pairwise_vector_feature_to_similarity_vals[vector_feat]
        
    return pairwise_data_df


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add('--data', #note that the required data structure is explained in repository README as well as the example.ipynb tutorial notebook
               default='data/example/',
               type=str,
               help='path to the data directory containing decisions.csv and subdirectory called features containing the various .csv files corresponding to the actor-level features')
    parser.add('--add_epsilon',
               action='store_true',
               help='Specify this to add small epsilon value to all elements for a feature vector - helpful in case feature vectors can be all zeroes for any individual actor (which can result in null values when computing pairwise similarity)')
    parser.add('--metric',
               default='Cosine',
               type=str,
               help='similarity metric to compute a score from pair of vector features for actors - currently, Cosine similarity is both the default and the only metric supported; implementation in utils.py')
    parser.add('--output',
               default='data/example/output/',
               type=str,
               help='path to save the output pairwise data file used to run the mixed effects model as well as record the removed actor pairs')
    
    args = parser.parse_args()
    add_epsilon = args.add_epsilon
    data_base_path = args.data
    metric = args.metric
    output_path = args.output
    
    decisions_df = pd.read_csv(Path(data_base_path) / 'decisions.csv', 
                               header = None)
    actor_ids = list(decisions_df[0])
    actor_ids = list(map(lambda x:str(x), actor_ids))
    decision_vals = decisions_df.iloc[:,1:].values
    
    print('Loading various actor-level features...')
    scalar_features_to_values, vector_features_to_values = load_feature_values_dicts(data_base_path)
    
    print('Computing pairwise-level data required for the codecision model...')
    pairwise_actor_ids, pairwise_codecision_agreement_rate, pairwise_scalar_feature_to_same_identity_vals, pairwise_vector_feature_to_similarity_vals, removed_pairs, indiv_actors1, indiv_actors2 = get_pairwise_data_elements(actor_ids, decision_vals, scalar_features_to_values, vector_features_to_values, metric, add_epsilon)
    
    pairwise_df = get_pairwise_dataframe(pairwise_actor_ids, pairwise_codecision_agreement_rate, pairwise_scalar_feature_to_same_identity_vals, pairwise_vector_feature_to_similarity_vals, indiv_actors1, indiv_actors2, metric)
    
    #store the pairs of legislators removed for perfect agreement or disagreement following Ringe et al. (2013) 
    f = open(Path(output_path) / "removed_pair_actor_ids.txt", 'w')
    for pair in removed_pairs:
        f.write(str(pair))
        f.write('\n')
    f.close()
    
    print('Storing pairwise-level data required for the codecision modeling...')
    #store the pairwise data output - used as input to mixed effects modeling in R
    pairwise_df.to_csv(Path(output_path) / "pairwise_data.csv", index=False)
    print('Done.')
    