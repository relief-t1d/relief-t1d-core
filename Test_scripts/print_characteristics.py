import platform
import socket
import os
import psutil
import tensorflow as tf
import sys

import subprocess

def print_nvidia_info():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        print("\n🎮 NVIDIA GPU Info:")
        print("-------------------------------")
        for line in result.decode().strip().split('\n'):
            name, mem, driver = map(str.strip, line.split(','))
            print(f"GPU Name        : {name}")
            print(f"Memory          : {mem}")
            print(f"Driver Version  : {driver}")
            print("-------------------------------")
    except FileNotFoundError:
        print("⚠️  'nvidia-smi' no encontrado: ¿Estás en una máquina con GPU NVIDIA y drivers instalados?")
    except Exception as e:
        print(f"⚠️  Error al ejecutar nvidia-smi: {e}")


def print_system_info():
    print("🖥️  System Info")
    print("-------------------------------")
    print(f"Hostname        : {socket.gethostname()}")
    print(f"OS              : {platform.system()} {platform.release()} ({platform.version()})")
    print(f"Architecture    : {platform.machine()}")
    print(f"CPU             : {platform.processor()}")
    print(f"CPU Cores       : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical")
    print(f"RAM             : {round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB")

    print("\n🐍 Python & Libraries")
    print("-------------------------------")
    print(f"Python Version  : {platform.python_version()}")
    print(f"TensorFlow      : {tf.__version__}")
    print(f"Executable Path : {sys.executable}")


print_system_info()

print_nvidia_info()
