from general_functions import functions, database_data
from data import loader
from general_functions.models import BasicModel


class Rms(BasicModel):
    def __init__(self, config_path, shot, model_name='rms'):
        super().__init__(config_path, shot, model_name)
        print(raw_data)

    def calculate_tags_rms(self, raw_data):
        data = raw_data['data']


def main():
    # test_shots_calculate()
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    shot = '1096327'
    Rms(config_path, shot)
    # test_shots_calculate()


