import os
import socket

import yaml


def get_server_port():
    port = 5050
    if os.path.exists('docker-compose.yml'):
        with open("docker-compose.yml", 'r') as f:
            docker_yml = yaml.safe_load(f)
            services = list(docker_yml['services'].keys())
            port = docker_yml['services'][services[0]]['ports'][0].split(':')[0]
    return port


def get_redis_port():
    port = 6379
    if os.path.exists('docker-compose-service-with-redis.yml'):
        with open("docker-compose-service-with-redis.yml", 'r') as f:
            docker_yml = yaml.safe_load(f)
            services = list(docker_yml['services'].keys())
            port = docker_yml['services'][services[-1]]['ports'][0].split(':')[0]
    return port


def create_configuration():
    config_dict = dict()
    if os.path.exists('config.yml'):
        with open("config.yml", 'r') as f:
            config_dict = yaml.safe_load(f)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    config_dict['service_host'] = s.getsockname()[0]
    s.close()

    config_dict['service_port'] = get_server_port()

    write_configuration(config_dict)


def write_configuration(config_dict):
    with open('config.yml', 'w') as config_file:
        for config_key, config_value in config_dict.items():
            config_file.write(f'{config_key}: {config_value}\n')


if __name__ == '__main__':
    # create_configuration()
    print(get_redis_port())
    print(get_redis_port())
