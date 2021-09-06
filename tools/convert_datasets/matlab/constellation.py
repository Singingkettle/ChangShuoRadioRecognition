import numpy as np


class Constellation(object):
    # matrix window info
    height_range = [-1, 1]
    width_range = [-1, 1]

    # parameters for converting sequence
    # data (2, N) to constellation matrix based on conv mode
    filter_size = [0.05, 0.02]
    filter_stride = [0.05, 0.02]

    @classmethod
    def get_filters(cls):
        filters = []
        for filter_size, filter_stride in zip(cls.filter_size, cls.filter_stride):
            filters.append([filter_size, filter_stride])

        return filters

    @classmethod
    def generate_by_filter(cls, data):

        constellations = []
        filters = []
        for filter_size, filter_stride in zip(cls.filter_size, cls.filter_stride):
            matrix_width = int((cls.width_range[1] - cls.width_range[0] - filter_size) / filter_stride + 1)
            matrix_height = int((cls.height_range[1] - cls.height_range[0] - filter_size) / filter_stride + 1)

            constellation = np.zeros((matrix_height, matrix_width))

            def axis_is(query_axis_x, query_axis_y):
                axis_x = query_axis_x // filter_stride
                axis_y = query_axis_y // filter_stride
                if axis_x * filter_stride + filter_size < query_axis_x:
                    position = [None, None]
                elif axis_y * filter_stride + filter_size < query_axis_y:
                    position = [None, None]
                else:
                    position = [int(axis_x), int(axis_y)]
                return position

            pos_list = map(axis_is, list(data[0, :]), list(data[1, :]))
            num_point = 0
            for pos in pos_list:
                if pos[0] is not None:
                    constellation[pos[0], pos[1]] += 1
                    num_point += 1
            constellations.append(constellation / num_point)
            filters.append([filter_size, filter_stride])

        return constellations, filters


if __name__ == '__main__':
    test = np.random.random((2, 1000000))

    Constellation.generate_by_filter(test)
