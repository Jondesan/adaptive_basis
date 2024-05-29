def print_object(obj):
    for property, value in vars(obj).items():
        print(property, ":", value)