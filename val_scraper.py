import requests
from bs4 import BeautifulSoup
import re
import csv


val_page = "https://www.amherst.edu/campuslife/housing-dining/dining/menu/2017-10-15/2017-10-21"
r = requests.get(val_page)
page = r.content
soup = BeautifulSoup(page, "html.parser")

meals = soup.find_all("div", class_="dining-menu-meal")

data = []
for meal in meals:
    date = meal.a['data-date']
    meal_type = meal.a['data-meal']
    content = ""
    
    breakfast_match = re.search('breakfast', meal_type.lower())

    if not breakfast_match:
        print("not breakfast, finding traditional")
        found = False
        tag = meal.div.div
        print(tag.prettify())
        while not found:
            traditional_match = re.search('traditional', tag.string.lower())
            if traditional_match:
                print("traditional found!")
                found = True
            else:
                tag = tag.next_sibling
        content = tag.next_sibling.string
        print(content)
    new = (date, meal_type, content)
    data.append(new)

with open('test_val2.csv', 'w') as f:
    writer = csv.writer(f, delimiter = ",")
    writer.writerow(["date", "type", "food"])
    for row in data:
        writer.writerow(row)
                