# def recurrence_copy_dict(data, results, keys):
#     if isinstance(keys, str):
#         data[keys] = results[keys]
#         return data
#     elif isinstance(keys, dict):
#         for k, v in keys.items():
#             data[k] = dict()
#             return recurrence_copy_dict(data[k], results[k], v)
#     elif isinstance(keys, list):
#         for k in keys:
#             recurrence_copy_dict(data, results, k)
#         return data
#
#
# data = dict()
#
# keys = ['a', 'b']
#
# results = dict(b=1, a=4)
#
#
# print(recurrence_copy_dict(data, results, keys))
# results = dict(a=dict(b=1, a=4))
# keys1 = [dict(a=['a', 'b'])]
#
# data = dict()
# print(recurrence_copy_dict(data, results, keys1))