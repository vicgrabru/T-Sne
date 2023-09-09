import csv, math
def read_csv(route):
    with open(route, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        columns = len(reader[0])-1
        entries = len(reader)
        matrix_size = math.sqrt(columns)
        digits = []

