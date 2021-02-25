from data_classes.args_data import ArgsData


def parse_args(parameters):
    arr_length = len(parameters)
    if arr_length == 0:
        return None

    args_data = ArgsData()
    if arr_length >= 1:
        args_data.root_data_path = parameters[0]
    if arr_length == 2:
        args_data.site_ids = parameters[1].split(",")

    return args_data
