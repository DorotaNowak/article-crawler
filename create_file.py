import csv

with open('articles.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Id", "Link"])
file.close()