from general_functions import functions, database_data
from data import loader, utils, processor
from general_functions.models import BasicModel


class Rms(BasicModel):
    def __init__(self, config_path, shot, model_name='rms'):
        super().__init__(config_path, shot, model_name)
        raw_data = self.load_raw_data()
        # print(raw_data)
        tags_rms = self.calculate_tags_rms(raw_data)
        avg_rms = self.calculate_directional_average_rms(tags_rms)
        print(tags_rms)
        print(avg_rms)
        # print(V_avg_rms)

    def calculate_tags_rms(self, raw_data, remove_pre='\\Vib'):
        tags_rms = {}
        data = raw_data['data']
        for tag in self.tags:
            key = utils.remove_prefix(tag, remove_pre) + '_rms'
            dataset = data[tag]['dataset']
            rms = processor.calculate_tag_rms(dataset)
            tags_rms[key] = rms
        return tags_rms

    @staticmethod
    def calculate_directional_average_rms(data):
        """
        计算纵场线圈与真空室上的传感器在R、phi、Z方向上的平均rms，并以特定格式返回结果。
        :param data: 包含rms值的字典
        :return: 返回两类位置（TF, V）在三个方向（R, phi, Z）上的平均rms值，按指定格式组织
        """
        F_sum = {'R': 0, 'phi': 0, 'Z': 0}
        V_sum = {'R': 0, 'phi': 0, 'Z': 0}
        F_count = {'R': 0, 'phi': 0, 'Z': 0}
        V_count = {'R': 0, 'phi': 0, 'Z': 0}

        for key, value in data.items():
            direction = None
            if 'R' in key:
                direction = 'R'
            elif 'phi' in key or 'PHI' in key:
                direction = 'phi'
            elif 'Z' in key:
                direction = 'Z'

            if direction is not None:
                if 'TF' in key:
                    F_sum[direction] += value
                    F_count[direction] += 1
                elif 'V' in key:
                    V_sum[direction] += value
                    V_count[direction] += 1

        # 计算平均值并构建结果字典
        result = {}
        for loc, sums, counts in [('TF', F_sum, F_count), ('V', V_sum, V_count)]:
            for direction in sums.keys():
                avg_key = f"{loc}_{direction}_rms"
                result[avg_key] = (sums[direction] / counts[direction]) if counts[direction] != 0 else 0

        return result



def main():
    # test_shots_calculate()
    # # 输入参数
    # config_path, name, shot = functions.get_input_params('rms')
    config_path = './config.yml'
    shot = '1103998'
    Rms(config_path, shot, model_name='rms')
    # test_shots_calculate()


if __name__ == '__main__':
    main()
