def get_variable_from_file(file_path, variable_name):
    global_namespace = {}
    with open(file_path, 'r') as file:
        exec(file.read(), global_namespace)
    return global_namespace.get(variable_name)