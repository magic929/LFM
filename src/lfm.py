import pandas as pd
from math import exp
import numpy as np
import sys
import pickle
import os

def get_negative_item(frame, user_id):
    print("get user %d negative item ---" % (user_id))
    userItems = list(set(frame[frame['userId'] == user_id]['movieId']))
    otherItems = list(set(frame[~frame['movieId'].isin(userItems)]['movieId'].values))
    # otherItems = [item for item in set(frame['movieId'].values) if item not in userItems]
    # itemCount = [len(frame[frame['movieId'] == item]['userId']) for item in otherItems]
    itemCount = frame[frame['movieId'].isin(otherItems)].groupby(['movieId']).count().sort_values(by='userId', ascending=False).head(len(userItems))
    # series = pd.Series(itemCount, index=otherItems)
    # series = itemCount.sort_values(ascending=False)[:len(userItems)]
    negativeItems = list(itemCount.index)
    return negativeItems


def get_postive_item(frame, user_id):
    series = frame[frame['userId'] == user_id]['movieId']
    positiveItems = list(series.values)
    return positiveItems


def init_user_item(frame, user_id=1):
    positiveItem = get_postive_item(frame, user_id)
    negativeItem = get_negative_item(frame, user_id)
    itemDict = {}
    for item in positiveItem: itemDict[item] = 1
    for item in negativeItem: itemDict[item] = 0
    return itemDict


def user_item_pool(frame, user_id):
    userItem = []
    for uid in user_id:
        itemDict = init_user_item(frame, uid)
        userItem.append({uid: itemDict})
        if len(userItem) >= 1000:
            return userItem
    
    return userItem


def init_para(user_id, item_id, class_count):
    arrayp = np.random.rand(len(user_id), class_count)
    arrayq = np.random.rand(class_count, len(item_id))
    p = pd.DataFrame(arrayp, columns=range(0, class_count), index=user_id)
    q = pd.DataFrame(arrayq, columns=item_id, index=range(0, class_count))

    return p, q


def init_model(frame, class_count):
    userId = list(set(frame['userId'].values))
    itemId = list(set(frame['movieId'].values))
    p, q = init_para(userId, itemId, class_count)
    if not os.path.exists("output/userItem-1000.pk"):
        userItem = user_item_pool(frame, userId)
        with open('output/userItem-1000.pk', 'wb') as f:
            pickle.dump(userItem, f, pickle.HIGHEST_PROTOCOL)
    else: 
        with open('output/userItem-1000.pk', 'rb') as f:
            userItem = pickle.load(f)
    return p, q, userItem


def laten_factor_model(frame, class_count, iter_count, alpha, lamb):
    p, q, userItem = init_model(frame, class_count)
    for step in range(0, iter_count):
        for user in userItem:
            for uid, samples in user.items():
                for item, rui in samples.items():
                    eui = rui - lfm_predict(p, q, uid, item)
                    for f in range(0, class_count):
                        print("step %d user %d class %d" % (step, uid, f))
                        p[f][uid] += alpha * (eui * q[item][f] - lamb * p[f][uid])
                        q[item][f] += alpha * (eui * p[f][uid] - lamb * q[item][f])
        alpha *= 0.9
    
    return p, q


def sigmod(x):
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    e = np.exp(x)
    return e / (e + 1.0)

def lfm_predict(p, q, user_id, item_id):
    p = np.mat(p.iloc[user_id].values)
    q = np.mat(q[item_id].values).T
    r = (p * q).sum()
    r = sigmod(r)

    return r


def recommend(frame, user_id, p, q, top_n=10):
    userItems = list(set(frame[frame['userId'] == user_id]['movieId']))
    # otherItems = [item for item in set(frame['movieId'].values) if item not in userItems]
    otherItems = list(set(frame[~frame['movieId'].isin(userItems)]['movieId'].values))
    predicts = [lfm_predict(p, q, user_id, item) for item in otherItems]
    series = pd.Series(predicts, index=otherItems)
    series = series.sort_values(ascending=False)[:top_n]
    return series


def recall(df_test, p, q):
    hit = 0
    all_item = 0
    df_userid = df_test['userId']
    df_userid = df_userid.drop_duplicates()
    for uid in df_userid:
        pre_item = recommend(df_test, uid, p, q)
        df_item = df_test.loc[df_test['userId'] == uid]
        true_item = df_item['movieId']
        for item, prob in pre_item.items():
            if item in true_item:
                hit += 1
        all_item += len(true_item)
    
    return hit/(all_item * 1.0)


def precision(df_test, p, q):
    hit = 0
    all_item = 0
    df_userid = df_test['userId']
    df_userid = df_userid.drop_duplicates()
    for uid in df_userid:
        pre_item = recommend(df_test, uid, p, q)
        df_item = df_test.loc[df_test['userId'] == uid]
        true_item = df_item['movieId']
        for item, prob in pre_item.items():
            if item in true_item:
                hit += 1
        all_item += len(pre_item)
    return hit/(all_item * 1.0)


if __name__ == '__main__':
    # start = time.clock()
    df_sample = pd.read_csv(sys.argv[1], names=['userId', 'movieId', 'rating', 'timestamp'], header=0)
    if sys.argv[2] == 'train':
        # df_sample = pd.read_csv(sys.argv[1], names=['userId', 'movieId', 'rating', 'timestamp'], header=0)
        p, q = laten_factor_model(df_sample, 5, 3, 0.02, 0.01)
        with open('output/para_p.pk', 'wb') as f:
            pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
        with open('output/para_q.pk', 'wb') as f:
            pickle.dump(q, f, pickle.HIGHEST_PROTOCOL)
    
    if sys.argv[2] == 'recommend':
        user_id = 1111
        print("user {} may like: ".format(user_id))
        with open('output/para_p.pk', 'rb') as f:
            p = pickle.load(f)
        with open('output/para_q.pk', 'rb') as f:
            q = pickle.load(f)
        result = recommend(df_sample, user_id, p, q)
        movies = pd.read_csv('./input/ml-25m/movies.csv')
        movies_list = movies[movies['movieId'].isin(result.index)]['title'].values
        print(movies_list)

    # print(recall(df_test, p, q))
    # print(precision(df_test, q, p))