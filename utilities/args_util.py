from data_classes.args_data import ArgsData


def parse_args(parameters):
    """
    parses the arguments supplied when executing the program
    :param parameters: list of arguments supplied when executing
    :return: object of type Args_Data consisting of
        a) string with data root folder
        b) list of strings with site id's (split at ",") if supplied
        If no parameters are supplied it returns None
    """
    arr_length = len(parameters)
    if arr_length == 0:
        return None

    args_data = ArgsData()
    if arr_length >= 1:
        args_data.root_data_path = parameters[0]
    if arr_length == 2:
        args_data.site_ids = parameters[1].split(",")

    return args_data
