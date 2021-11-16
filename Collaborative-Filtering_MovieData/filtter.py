'''
@company:NJU
@author:wjw
@contact:1508417398@qq.com
'''
import pandas as pd
import numpy as np

def data_spilt(data_path, x=0.8, random=False):
    print("开始切分数据集...")
    dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
    testset_index = []
    ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"] == uid).dropna()
        if random:
            # 因为不可变类型不能被shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)  # 打乱列表
            _index = round(len(user_rating_data) * x)
            testset_index += list(index[_index:])
        else:
            _index = round(len(user_rating_data) * x)
            testset_index += list(user_rating_data.index.values[_index:])

    testset = ratings.loc[testset_index]
    trainset = ratings.drop(testset_index)
    print("完成数据集切分...")
    return trainset, testset,ratings_matrix
trainset,testset,ratings_matrix= data_spilt('C:/Users/lenovo/Desktop/ml-latest-small/ratings.csv')
print(testset)
#计算用户之间相似度,相似矩阵
user_similar = ratings_matrix.T.corr()
def predict(uid, iid, ratings_matrix, user_similar):
    try:
        print("开始预测用户<%d>对电影<%d>的评分..." % (uid, iid))
        similar_users = user_similar[uid].drop([uid]).dropna()
        similar_users = similar_users.where(similar_users > 0).dropna()
        if similar_users.empty is True:
            raise Exception("用户<%d>没有相似的用户" % uid)
        ids = set(ratings_matrix[iid].dropna().index) & set(similar_users.index)
        finally_similar_users = similar_users.loc[list(ids)]
        numerator = 0
        denominator = 0
        for sim_uid, similarity in finally_similar_users.iteritems():
            sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
            sim_user_rating_for_item = sim_user_rated_movies[iid]
            numerator += similarity * sim_user_rating_for_item
            denominator += similarity
        predict_rating = numerator / denominator
        print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
        return round(predict_rating, 2)
    except Exception as e:
        print(e)
        return 0
def predict_all(uid, ratings_matrix, user_similar):
    item_ids = ratings_matrix.columns
    for iid in item_ids:
        rating = predict(uid, iid, ratings_matrix, user_similar)
        yield uid, iid, rating
def top_k_rs_result(userId,k):
    results = predict_all(userId, ratings_matrix, user_similar)
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]
def accuray(uids,mids,real_ratings,pre_ratings):
    length = 0
    _rmse_sum = 0
    for i in range(len(uids)):
        uid=uids[i]
        mid=mids
        real_rating=real_ratings[i]
        pre_rating=pre_ratings[i]
        length += 1
        _rmse_sum += (pre_rating-real_rating)**2
    return round(np.sqrt(_rmse_sum/length),4)
if __name__ == '__main__':
    from pprint import pprint
    uids=[]
    mids=[]
    real_ratings=[]
    pre_ratings=[]
    for i in range(len(testset)):
        uid=list(testset['userId'])[i]
        uids.append(uid)
        mid=list(testset['movieId'])[i]
        mids.append(mid)
        real_ratings.append(list(testset['rating'])[i])
        pre_ratings.append(predict(uid,mid,ratings_matrix,user_similar))
    import csv
    f = open('movie_1.csv', 'w', encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['userId', 'movieId'])
    for i in range(610):
        result = top_k_rs_result(i+1,5)
        for j in range(len(result)):
            csv_writer.writerow([i+1,result[j]])
    #标准误差
    print(accuray(uids,mids,real_ratings,pre_ratings))


