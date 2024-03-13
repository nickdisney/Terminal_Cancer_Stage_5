import subprocess

class SoftwareController:
    def __init__(self):
        pass

    def execute_command(self, command):
        try:
            output = subprocess.check_output(command, shell=True, universal_newlines=True)
            return output
        except subprocess.CalledProcessError as e:
            return f"Error: {e}"