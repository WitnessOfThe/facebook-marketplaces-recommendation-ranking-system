# %%
import faiss                   # make faiss available
import json
import pickle
import numpy as np
def get_dic_to_numpy_array():
    with open('image_embeddings.json') as json_file:
        dic = json.load(json_file)
    matrix = np.empty([0,13])
    for key in dic.keys():
        matrix = np.vstack([matrix,np.array(dic[key], dtype=np.float32)])
    return matrix

def get_FAISS_index():
    d  = 13
    xb = get_dic_to_numpy_array() 
    nb = len(xb)

    index_flat = faiss.IndexFlatL2(d)  # build a flat (CPU) index
    index_flat.add(xb)   
    return index_flat
    
# %%
if __name__ == '__main__':

    index = get_FAISS_index()
    faiss.write_index(index,'FAISS_index.pkl')
