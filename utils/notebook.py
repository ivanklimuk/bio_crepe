import os


def set_env_vars(env_file):
    with open(env_file, 'r') as file:
        for line in file.readlines():
            if line is not '\n':
                key, value = line.strip().split('=')
                os.environ[key] = value