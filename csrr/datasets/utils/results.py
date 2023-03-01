
def list_dict_to_dict_list(results):
    assert isinstance(results, list), 'the results must be a list'
    assert isinstance(results[0], dict), 'items in results must be a dict, whcih all have the keys'

    results = {k: [dic[k] for dic in results] for k in results[0]}

    return results
