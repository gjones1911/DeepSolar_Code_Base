"""                  This file will load a set of given SVI csv files and stack them into one big file"""

import numpy as np
import pandas as pd



AL_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Alabama.csv')
AZ_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Arizona.csv')
CA_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\California.csv')
GA_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Georgia.csv')
KY_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Kentucky.csv')
MA_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Massachusetts.csv')
MS_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Mississippi.csv')
NY_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\NewYork.csv')
NC_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\NorthCarolina.csv')
TN_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Tennessee.csv')
TX_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Texas.csv')
UT_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Utah.csv')
VA_df = pd.read_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI\Virginia.csv')
colsmn = AL_df.columns.values.tolist()

new_df = AL_df.values

states = [AL_df, AZ_df, CA_df, GA_df, KY_df, MA_df, MS_df, NY_df, NC_df, TN_df, TX_df, UT_df, VA_df,]
states = [AZ_df, CA_df, GA_df, KY_df, MA_df, MS_df, NY_df, NC_df, TN_df, TX_df, UT_df, VA_df,]
"""        NOW STACK AND SAVE THEM   """
for st in states:
    new_df  = np.vstack((new_df, st.values))

new_df = pd.DataFrame(new_df, columns=colsmn)
new_df.to_csv(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI/SVI_.csv')
# new_df.to_excel(r'C:\Users\gjone\DeepSolar_Code_Base\__Data\__SVI/SVI_.xlsx')
