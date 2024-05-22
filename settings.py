
def add(var):
    globals()[str(var[0])] = var

def return_globals():
    return globals()
