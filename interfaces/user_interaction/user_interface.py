class UserInterface:
    def __init__(self):
        pass

    def get_user_input(self):
        user_input = input("User: ")
        return user_input

    def display_output(self, output):
        print(f"Assistant: {output}")