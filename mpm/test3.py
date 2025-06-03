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


def get_disruption_shots(connection_str, collection, min_shot: int, max_shot: int):
    db = MetaDB(connection_str, collection)
    # %%
    #  find all the shot with shot_no in range [10000, 20000] && [IP, BT] tags available && is disruption
    shot_list = [shot for shot in range(min_shot, max_shot)]
    disruption_shots = db.query_valid(
        shot_list=shot_list,
        label_true=["IsDisrupt", "ip", "bt"],
    )
    return disruption_shots


def get_not_disruption_shots(connection_str, collection, min_shot: int, max_shot: int):
    db = MetaDB(connection_str, collection)
    # %%
    #  find all the shot with shot_no in range [10000, 20000] && [IP, BT] tags available && is disruption
    shot_list = [shot for shot in range(min_shot, max_shot)]
    not_disruption_shots = db.query_valid(
        shot_list=shot_list,
        label_true=["ip", "bt"],
        label_false=['IsDisrupt']
    )
    return not_disruption_shots


def get_is_disruption(connection_str, collection, target_shot: int):
    if get_disruption_shots(connection_str, collection, target_shot+1 , target_shot):
        return True
    elif get_not_disruption_shots(connection_str, collection, target_shot+1 , target_shot):
        return  False
    else:
        return None

# db = MetaDB(connection_str, collection)
#
# # %%
# #  find all the shot with shot_no in range [10000, 20000] && [IP, BT] tags available && is disruption
# shot_list = [shot for shot in range(1054200, 1054300)]
# complete_disruption_shots = db.query_valid(
#     shot_list=shot_list,
#     label_true=["IsDisrupt", "ip", "bt"]
# )
# print(complete_disruption_shots)
# print(len(complete_disruption_shots))