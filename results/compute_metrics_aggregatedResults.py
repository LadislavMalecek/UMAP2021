from pathlib import Path
import sys, os
from typing import Dict, List, NamedTuple, Tuple
import numpy as np
from collections import defaultdict
import time



# fold, directory
def get_folds(data_dir: str) -> List[Tuple[int, str]]:
    folds = []
    for dir in [f for f in Path(data_dir).iterdir() if f.is_dir()]:
        dir_name = os.path.basename(dir)
        if str(dir_name).isnumeric():
            folds.append((int(dir_name), str(dir)))
    folds.sort()
    return folds

# returns 2d numpy array where 1. index is userId and 2. index is itemId, values are float ratings
def load_data(data_dir: str, fold: int) -> np.ndarray:
    return np.load(os.path.join(data_dir, str(fold), "mf_data.npy"))

class Group(NamedTuple):
    id: int
    members: List[int]

# group data must be in file formated with groupId, userid1, userid2...
# separated by tabs
def load_group_data(data_dir: str, group_type: str, group_size: int) -> List[Group]:
    groups = []
    filename = group_type + "_group_" + str(group_size)
    path = os.path.join(data_dir, filename)
    with open(path) as group_file:
        lines = group_file.readlines()
        for line in lines:
            items = line.replace('\n', '').split("\t")
            items = list(map(int, items))
            groups.append(Group(items[0], items[1:]))
            if len(items) < group_size + 1:
                raise Exception("Group file invalid: " + path)
    return groups
    
    
    
def get_recommendation_files(data_dir: str, fold: int, group: str, group_size: int) -> List[str]:
    rec_path = os.path.join(data_dir, str(fold), group, str(group_size)) 
    return list([str(f) for f in Path(rec_path).iterdir() if f.is_file()])

class AlgRecommendations(NamedTuple):
    alg_name: str
    # dict indexed by groupId
    group_recommendations: Dict[int, List[int]] = {} 


# items are sorted from best to worst
# returns list of tuples where first is the agreg name and second is dictionary of recommendations indexed by group id
def load_agregated_recommendations(data_dir: str, fold: int, group: str, group_size: int) -> List[AlgRecommendations]:
    files = get_recommendation_files(data_dir, fold, group, group_size)
    returnList = []
    for file in files:
        recommendationsMap = defaultdict(list) 
        with open(file) as recommendation_file:
            lines = recommendation_file.readlines()
            for line in lines:
                items = line.replace('\n', '').split("\t")[:2]
                items = list(map(int, items))
                group_id = items[0]
                recommendationsMap[group_id].append(items[1])
        alg_name = os.path.basename(file)
        returnList.append(AlgRecommendations(alg_name, recommendationsMap))
    return returnList

#calculates discounted cumulative gain on the array of relevances
def calculate_dcg(values):
    values = np.array(values)
    if values.size: #safety check
        return np.sum(values / np.log2(np.arange(2, values.size + 2)))
    return 0.0    

#order items of user, cut best topk_size, calculate DCG of the cut
#test_data = uidxoid matrix of ratings
#topk_size = volume of items per user on which to calculate IDCG
#return dictionary {userID:IDCG_value}
def calculate_per_user_IDCG(test_data, topk_size):
    users = range(test_data.shape[0])
    idcg_per_user = {}
    for user in users:        
        per_user_items = test_data[user] 
        sorted_items = np.sort(per_user_items)[::-1]
        sorted_items = sorted_items[0:20]
        
        idcg = calculate_dcg(sorted_items)
        idcg_per_user[user] = idcg
        
        #print(sorted_items)
        #print(idcg)
        #exit()
        
    return idcg_per_user
        
    

class Result(NamedTuple):
    alg: str
    metric: str
    value: float

  

def compute_metrics(test_data: np.ndarray, groups: List[Group], alg_data: AlgRecommendations) -> List[Result]:
    # test_data are triplets: user_id, item_id, and rating
    #LP: test data is matrix user_id x item_id !!!!!! a ja si rikal, jakto ze ti to prirazeni funguje...
    idcg_per_user = calculate_per_user_IDCG(test_data, 20)
    
    
    avg_rating = []
    min_rating = []
    minmax_rating = []
    std_rating = []
    
    avg_nDCG_rating = []
    min_nDCG_rating = []
    minmax_nDCG_rating = []
    std_nDCG_rating = []
        
    for group in groups:
        group_users_sum_ratings = []
        group_users_ndcg_ratings = []
        group_id = group.id 
        rec_for_group = alg_data.group_recommendations[group_id]
        for group_user_id in group.members:
            user_sum = 0.0
            user_list = []
            for item_id in rec_for_group:
                rating = test_data[group_user_id, item_id]
                #print(group_user_id, item_id, rating)
                #print(type(test_data))
                #print(test_data.shape)
                #print(test_data[group_user_id])
                #exit()
                user_sum += rating
                user_list.append(rating)
            ndcg = calculate_dcg(user_list) / idcg_per_user[group_user_id]   
            
            group_users_sum_ratings.append(user_sum)
            group_users_ndcg_ratings.append(ndcg)
        #TODO: quick&dirty code - consider revising   
        
        group_users_mean_ratings = [i/len(rec_for_group) for i in group_users_sum_ratings] 
        avg_rating.append(np.average(group_users_mean_ratings)) 
        min = np.min(group_users_mean_ratings)
        min_rating.append(min) 
        max = np.max(group_users_mean_ratings)
        minmax_rating.append(0.0 if max == 0.0 else min/max)
        std_rating.append(np.std(group_users_mean_ratings)) 
        
        avg_nDCG_rating.append(np.average(group_users_ndcg_ratings)) 
        min = np.min(group_users_ndcg_ratings)
        min_nDCG_rating.append(min) 
        max = np.max(group_users_ndcg_ratings)
        minmax_nDCG_rating.append(0.0 if max == 0.0 else min/max)
        std_nDCG_rating.append(np.std(group_users_ndcg_ratings))         
        
    results = []
    results.append(Result(alg_data.alg_name, "AR_avg", np.average(avg_rating)))
    results.append(Result(alg_data.alg_name, "AR_min", np.average(min_rating)))
    results.append(Result(alg_data.alg_name, "AR_min/max", np.average(minmax_rating)))
    results.append(Result(alg_data.alg_name, "AR_std", np.average(std_rating)))
    
    results.append(Result(alg_data.alg_name, "nDCG_avg", np.average(avg_nDCG_rating)))
    results.append(Result(alg_data.alg_name, "nDCG_min", np.average(min_nDCG_rating)))
    results.append(Result(alg_data.alg_name, "nDCG_min/max", np.average(minmax_nDCG_rating)))
    results.append(Result(alg_data.alg_name, "nDCG_std", np.average(std_nDCG_rating)))    
    return results


def process_fold(groups: List[Group], data_dir: str, fold: int, group: str, group_size: int) -> List[Result]:
    algs_data = load_agregated_recommendations(data_dir, fold, group, group_size)
    test_data = load_data(data_dir, fold)
    results = []
    for alg_data in algs_data:
        results.extend(compute_metrics(test_data, groups, alg_data))
    #for result in results:
    #    print(result)
    return results

def main(data_folder, group_type, group_size):
    print(data_folder, group_type, group_size)
    folds = get_folds(data_folder)
    groups: List[Group] = load_group_data(data_folder, group_type, int(group_size))
    
    results = []
    for fold, _ in folds:
        results.extend(process_fold(groups, data_folder, fold, group_type, int(group_size)))

        
    algs = set(map(lambda x:x.alg, results))
    metrics = list(set(map(lambda x:x.metric, results)))
    print(metrics)
    metrics.sort()
    print(metrics)
    res = "alg,group_type,group_size" + "," + ",".join(metrics)+"\n"
    for alg in algs:
        values = [alg, group_type, str(group_size)]
        for metric in metrics:
            value = np.average([v.value for v in results if v.alg == alg and v.metric == metric])
            value = round(value,3)
            values.append(str(value))
        res += ",".join(values)+"\n"
    return res

        

if __name__ == "__main__":
    for group_type in ["sim", "div", "random"]:
        for group_size in ["2","3","4","8"]:
            f = open("results/result_"+group_type+"_"+group_size,"w")
            results = main("data/ml1m", group_type, group_size)
            
            f.write(results)

    #args = sys.argv[1:]
    #print(args)
    #main(args[0], args[1], args[2])
    #main("data/ml1m", "sim", "2")