#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import time

from multiprocessing import Process, Queue, Lock, Manager

from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.metrics import dtw_subsequence_path

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL) # ignore sigpipe


# In[2]:


dataset_folder = "../all"
metadata_filename = "training_set_metadata.csv"
data_filename = "training_set.csv"
get_ipython().system('ls {dataset_folder}')


# In[3]:


metadata = pd.read_csv("{}/{}".format(dataset_folder, metadata_filename))
metadata.head()


# In[4]:


object_ids = metadata.object_id.values
print("{} objects to predict".format(object_ids.size))


# In[5]:


def read_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data


# In[6]:


xg_model = read_pickle("xg_model")
shapelets = read_pickle("shapelets")
print(shapelets.shape)


# In[7]:


classes = np.array([ 6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99])
print("classes: ", classes)


# In[8]:


def normalize(ts, ts_err):
    ts /= (ts_err + 1) # +1 to avoid zero division
    ts = np.nan_to_num(ts)
    ts = TimeSeriesScalerMinMax().fit_transform(ts)
    return ts


# In[9]:


def distanceToShapelet(ts, shapelet):
    path, dist = dtw_subsequence_path(shapelet, ts)
    return dist


# In[10]:


def distanceToShapelets(ts, shapelets=shapelets):
    return [distanceToShapelet(ts, shapelet) for shapelet in shapelets]


# In[11]:


step = 1
nb_of_passband = 6
max_mjd = 1094
max_nb_of_processes = 3
print(nb_of_passband, max_mjd, sep="\n")


# In[12]:


def format_time_series(data):
    X = []
    X_err = []
    s0 = int(data.shape[0] / nb_of_passband)
    x, x_err = [], []
    for p in range(nb_of_passband):
        x = np.append(x, data[data.passband==p].flux.values)
        x_err = np.append(x_err, data[data.passband==p].flux_err.values)
    X.append(x.reshape(s0, nb_of_passband))
    X_err.append(x.reshape(s0, nb_of_passband))
    return np.array(X), np.array(X_err)


# In[13]:


def mjdToPredict(mjdToExclude, minMjd, maxMjd, step):
    return np.array(list(set(range(minMjd, maxMjd+1, step)) - set(mjdToExclude)))


# In[14]:


def fill_missing_flux(objectDf, maxMjd, step):
    objectDf.mjd = objectDf[["mjd"]].astype(np.int64)
    objectDf.mjd = objectDf.mjd - objectDf.mjd.min()
    instance = objectDf.groupby(["passband", "mjd"]).mean().reset_index()
    meanByPassband = instance.groupby("passband").mean()[["flux", "flux_err"]]
    frames = [instance]
    for p in range(6):
        toExclude = instance[instance.passband==p].mjd
        toPredict = mjdToPredict(toExclude, 0, maxMjd, step)
        nb = len(toPredict)
        frames.append(pd.DataFrame({
            'mjd': toPredict,
            'flux': [meanByPassband.loc[p].flux]*nb,
            'flux_err': [meanByPassband.loc[p].flux_err]*nb,
            'detected': [-1] * nb,
            'passband': [p]*nb,
            'object_id': [objectDf.object_id.values[0]]*nb
        }))
    return pd.concat(frames, sort=False)


# In[15]:


def predict_object(test_df):
    instance = fill_missing_flux(test_df, max_mjd, step)
    ts, ts_err = format_time_series(instance)
    ts_norm = normalize(ts, ts_err)
    dist_vec = distanceToShapelets(ts_norm[0], shapelets)
    preds = xg_model.predict_proba([dist_vec])[0]
    preds = np.concatenate(([test_df.object_id.values[0]], preds, [0]))
    return preds


# In[16]:


def producer_task(queue):
    data = pd.read_csv("{}/{}".format(dataset_folder, data_filename))
    for oid in object_ids[:20]:
#         print("Object {} produced".format(oid))
        object_df = data[data.object_id == oid]
        queue.put(object_df)


# In[17]:


def consumer_task(queue, lock, submission): 
    # Run indefinitely
    while True:
        # If the queue is empty, queue.get() will block until the queue has data
        object_df = queue.get()
        preds = predict_object(object_df)
#         print("Object {} consumed".format(preds[0]))
        # Synchronize access to the console
        with lock:
            submission.append(preds)


# In[18]:


def predict_test_set(predictions, nb_lines, max_nb_of_processes = 2):
    object_queue = Queue()

    lock = Lock()

    consumers = []

    producer = Process(target=producer_task, args=(object_queue,))

    for i in range(max_nb_of_processes):
        p = Process(target=consumer_task, args=(object_queue, lock, predictions))

        # This is critical! The consumer function has an infinite loop
        # Which means it will never exit unless we set daemon to true
        p.daemon = True
        consumers.append(p)

    producer.start()

    for c in consumers:
        c.start()

    len_preds = len(predictions)
    while len_preds < nb_lines:
        len_preds = len(predictions)
        print("{} / {} -> {} %".format(len_preds, nb_lines, np.round(len_preds * 100 / nb_lines if len_preds > 0 else 0)))
        time.sleep(2)
        
    print("{} %".format(np.round(len_preds * 100 / nb_lines if len_preds > 0 else 0)))
    print("Producer finished")


# In[19]:


get_ipython().run_cell_magic('time', '', 'manager = Manager()\n\npredictions = manager.list()\n\npredict_test_set(predictions, 20, max_nb_of_processes=max_nb_of_processes)')


# In[20]:


columns = ["object_id"]
columns.extend([f"class_{c}" for c in classes])
submission = pd.DataFrame(columns = columns, data=list(predictions))
submission.object_id = submission.object_id.astype(np.int32)
print(submission.shape)


# In[21]:


submission.to_csv("submission.csv", index=False)

