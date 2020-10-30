# coding=utf-8
import numpy as np
import pandas as pd
martix_ref=np.random.normal(loc=0.0,scale=1.0,size=(50,20))
martix_ref=pd.DataFrame(martix_ref)
martix_ref=martix_ref*(martix_ref>0)
u,sim,v=np.linalg.svd(martix_ref)
