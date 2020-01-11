#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:45:30 2019

@author: Soma Siddhartha
"""

import numpy as np
import torch
#%%
dset = np.random.rand(5000);
dset = dset > 0.5
#%%
parallel_datasets = []
for i in range(len(dset)):
    ind = np.ones(5000) == 1
    ind[i] = False
    parallel_datasets.append(dset[ind])
#%%
def create_db_and_parallels(size):
    db = np.random.rand(size)
    print(db)
    db = db > 0.5
    parallel_datasets = []
    
    for i in range(size):
        ind = np.ones(size) == 1
        ind[i] = False
        parallel_datasets.append(db[ind])
    
    return db, parallel_datasets
#%%
db, pdbs = create_db_and_parallels(20)
#%%
def query(db):
    return db.astype('float64').mean()
#%%
orig_db_result = query(db)
#%%
max_dist = 0
for pdb in pdbs:
    pd_result = query(pdb)
    
    db_dist = np.abs(orig_db_result - pd_result)
    if max_dist < db_dist:
        max_dist = db_dist
#%%
# The maximum amount that the query changes when we remove one individual from the database is known as L1 sensitivity.
def sensitivity(query, size):
    """
    Returns the sensitivity for the given query for the given database size.
    """
    db, pdbs = create_db_and_parallels(size)
    orig_db_result = query(db)
    print(orig_db_result)
    max_distance = 0
    for pdb in pdbs:
        pdb_result = query(pdb)
        pdb_dist = np.abs(orig_db_result-pdb_result)
        if max_distance < pdb_dist:
            max_distance = pdb_dist
    return max_distance
#%%
sensitivity(query1, 10)    
#%%
"""  Calculate L1 Sensitivity for Threshold """
#%%
def query1(db, threshold=5):
    return (db.sum() - threshold)
#%%
for i in range(10):
    sens_f = sensitivity(query1, 10)
    print(sens_f)
#%%    
""" Differencing Attack """
#%%
""" Local and Global Differential Privacy 

Local Differential Privacy : Adds noise to function data points. ( each individual data point ). Users are most protected here!
Global Differential Privacy : Adds noise to the output of the query on the database. Database contains all of the private info.Only for 'trusted curator'! 

"""
def query_local(db):
     db, _ = create_db_and_parallels(10)
     true_result = torch.mean(db.float())
     first_coin_flip = (torch.rand(len(db)) > 0.5).float()
     second_coin_flip = (torch.rand(len(db)) > 0.5).float()
     augmented_database = db * first_coin_flip + (1-first_coin_flip)*second_coin_flip
     db_result = torch.mean(augmented_database.float()) * 2 - 0.5
     return true_result, db_result




























