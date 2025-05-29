# %%
from jddb.meta_db import MetaDB

# %% connect to the MetaDB
connection_str = {
    "host": "www.limfx.pro",
    "port": 24007,
    "database": "DDB",
    "username": "DDBUser",
    "password": "tokamak!"
}

collection = "tags"

db = MetaDB(connection_str, collection)

# %%
#  find all the shot with shot_no in range [10000, 20000] && [IP, BT] tags available && is disruption
shot_list = [shot for shot in range(1054200, 1054300)]
complete_disruption_shots = db.query_valid(
    shot_list=shot_list,
    label_true=["IsDisrupt", "ip", "bt"]
)
print(complete_disruption_shots)
print(len(complete_disruption_shots))