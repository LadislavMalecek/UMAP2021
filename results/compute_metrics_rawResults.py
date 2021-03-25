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
    
class GroupWeights(NamedTuple):
    id: int
    members: List[float]    

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
    
# group data must be in file formated with groupId, weight1, weight2...
# separated by tabs
def load_group_weights_data(data_dir: str, group_type: str, group_size: int) -> List[Group]:
    groups = []
    filename = group_type + "_group_" + str(group_size)+"_weights"
    path = os.path.join(data_dir, filename)
    with open(path) as group_file:
        lines = group_file.readlines()
        for line in lines:
            items = line.replace('\n', '').split("\t")
            items = list(map(float, items))
            groups.append(GroupWeights(items[0], items[1:]))
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
    whitelist = ['GFAR', '_AVG', 'FuzzyDHondtDirectOptimize_1', 'GreedyLM',  'FuzzyDHondt_1',  'SPGreedy',  'fai',  'xpo']
    blacklist = "rec_rel"

    files = get_recommendation_files(data_dir, fold, group, group_size)
    r_files = []
    for file in files:
        for item in whitelist:
            if item in file and blacklist not in file:
                r_files.append(file)
    #print(r_files)
    #exit()    
    
    returnList = []
    for file in r_files:
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
    group_id: str
    user_id: int
    metric: str
    result: float


  

def compute_metrics(fold, test_data: np.ndarray, groups: List[Group],  alg_data: AlgRecommendations) -> List[Result]:
    # test_data are triplets: user_id, item_id, and rating
    #LP: test data is matrix user_id x item_id !!!!!! a ja si rikal, jakto ze ti to prirazeni funguje...
    idcg_per_user = calculate_per_user_IDCG(test_data, 20)
    
    
    results = []
    
    i = 0    
    for group in groups:
        #print(single_group_weights)
        group_users_sum_ratings = []
        group_users_ndcg_ratings = []
        group_id = group.id 
        rec_for_group = alg_data.group_recommendations[group_id]
        if len(rec_for_group) >0:
          j = 0
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
              j += 1
              
          group_users_mean_ratings = [i/len(rec_for_group) for i in group_users_sum_ratings] 
          
  
          for k in range(len(group_users_mean_ratings)):
              results.append(Result(alg_data.alg_name, str(group_id)+"_"+str(fold), group.members[k], "AR", group_users_mean_ratings[k])   )
              results.append(Result(alg_data.alg_name, str(group_id)+"_"+str(fold), group.members[k], "nDCG", group_users_ndcg_ratings[k])   )
         
          i += 1
        
    return results


def process_fold(groups: List[Group],  data_dir: str, fold: int, group: str, group_size: int) -> List[Result]:
    algs_data = load_agregated_recommendations(data_dir, fold, group, group_size)
    #print([i[0] for i in algs_data])
    #exit()
    test_data = load_data(data_dir, fold)
    results = []
    for alg_data in algs_data:
        results.extend(compute_metrics(fold, test_data, groups,  alg_data))
    #for result in results:
    #    print(result)
    return results

def main(data_folder, group_type, group_size):
    print(data_folder, group_type, group_size)
    folds = get_folds(data_folder)
    groups: List[Group] = load_group_data(data_folder, group_type, int(group_size))
    #group_weights: List[GroupWeights] = load_group_weights_data(data_folder, group_type, int(group_size))
    
    results = []
    for fold, _ in folds:
        results.extend(process_fold(groups,  data_folder, fold, group_type, int(group_size)))

        
    algs = set(map(lambda x:x.alg, results))
    metrics = set(map(lambda x:x.metric, results))
    res = ""
    for result in results:
        result = [str(i) for i in result]
        res += ",".join(result)+"\n"
    return res

        

if __name__ == "__main__":
    f = open("results/result_raw","w")
    res = "alg,group_id,user_id,metric,result\n"
    f.write(res)
    #for group_type in ["sim", "div", "random"]:
    #    for group_size in ["2","3","4","8"]:
    for group_type in ["sim", "div"]:
        for group_size in ["2","3","4","8"]:
            f2 = open("results/resultRaw_"+group_type+"_"+group_size,"w")
             
            results = main("data/ml1m", group_type, group_size)            
            f.write(results)
            f2.write(results)
            #exit()

    #args = sys.argv[1:]
    #print(args)
    #main(args[0], args[1], args[2])
    #main("data/ml1m", "sim", "2")