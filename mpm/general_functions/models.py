from general_functions import functions
from general_functions.database_data import connect_mongodb_database
from data import loader


class BasicModel:
    def __init__(self, config_path, shot, model_name):

        self.date_time, self.start_axis, self.start_sensor, self.is_running, self.temp = None, None, None, None, None
        self.shot, self.model_name, self.config_path = shot, model_name, config_path
        self.config, _ = functions.load_yaml(self.config_path)
        self.data_source, self.output_path, self.tags = (self.config['data_source'], self.config['Inference_path'],
                                                            self.config['tags'])
        # self.config['db']['collection'] = functions.replace_ball_mill_name(self.config['db']['collection'],)

        self.collection_name = self.config['db']['collection']
        self.client, self.db = connect_mongodb_database(self.config['db']['connection'], self.config['db']['db_name'])
        # raw_data = loader.get_raw_data(self.data_source, self.shot)

    def load_raw_data(self):
        return loader.get_raw_data(self.data_source, self.shot)

    def create_output_folder(self):
        output_path = functions.create_output_folder(self.config['Inference_path'], self.shot)
        return output_path


    # def confirm_threshold_exceeded(self, single_sensor, sensor_threshold):
    #     """
    #     确定rms与温度是否超出阈值，并且实现超出高级阈值会覆盖超出低级阈值
    #     :param single_sensor: 计算得到的异常指标
    #     :param sensor_threshold: 阈值
    #     :return:
    #     """
    #     for key, threshold in sensor_threshold.items():
    #         threshold = float(threshold)
    #         sensor_index = key.split('_')[2] + "_" + self.model_name
    #         alarm_index = key.split('_')[2] + "_" + self.model_name + "_alarm"
    #         # if single_sensor[sensor_index] >= threshold and key.count('h') > single_sensor[alarm_index]:
    #         #     single_sensor[alarm_index] = key.count('h')
    #         try:
    #             if single_sensor[sensor_index] >= threshold and key.count('h') > single_sensor[alarm_index]:
    #                 single_sensor[alarm_index] = key.count('h')
    #         except Exception as e:
    #             single_sensor[alarm_index] = 0
    #             print(e)
    #     return single_sensor
    #
    # def get_alarm(self, sensor, single_sensor):
    #     """
    #     更新的alarm部分
    #     :param sensor:传感器名称
    #     :param single_sensor:
    #     :return:
    #     """
    #     threshold_config, _, _ = functions.get_threshold_config(self.config['threshold_config_path'], self.name)
    #     # window, on, off = alarm_config['window'], alarm_config['on'], alarm_config['off']
    #     sensor_threshold = threshold_config['sensors_threshold'][sensor]
    #     # print(sensor_threshold, single_sensor)
    #     single_sensor_rms = self.confirm_threshold_exceeded(single_sensor, sensor_threshold)
    #     return single_sensor
    # # def get_is_running_raw_data(self):
