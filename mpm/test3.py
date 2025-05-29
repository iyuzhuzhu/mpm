# %%
from jddb.meta_db import MetaDB



# %% connect to the MetaDB
connection_str = {
    "host": 'www.limfx.pro',
    "port": 24007,
    "database": "JDDB"
}
#
# connection_str = {
#     "host": "localhost",
#     "port": 27017,
#     "database": "JDDB"
# }

collection = "Labels"

db = MetaDB(connection_str, collection)

# %%
#  find all the shot with shot_no in range [10000, 20000] && [IP, BT] tags available && is disruption
shot_list = [shot for shot in range(1103600, 1103900)]
complete_disruption_shots = db.query_valid(
    shot_list=shot_list,
    label_true=["IsDisrupt", "ip", "bt"]
)
print(complete_disruption_shots)
print(len(complete_disruption_shots))