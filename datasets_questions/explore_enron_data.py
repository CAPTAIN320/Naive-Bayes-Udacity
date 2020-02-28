#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

# How many data points (people)?
print("No. of people/datasets = ", len(enron_data))

# For each person, how many features are available?
print("No. of features per person = ", len(enron_data["SKILLING JEFFREY K"]))

# How many POIs are there in the E+F dataset?
print ("No. of POIs = ", len(dict((key, value) for key, value in enron_data.items() if value["poi"] == True)))

# How many POIs are there in total?
poi_reader = open('../final_project/poi_names.txt', 'r')
poi_reader.readline() # skip url
poi_reader.readline() # skip blank line

poi_count = 0
for poi in poi_reader:
	poi_count += 1

print("No. of TOTAL POIs = ", poi_count)

# What is the total value of the stock belonging to James Prentice?
print("Total stock value of James Prentice = ", enron_data["PRENTICE JAMES"]["total_stock_value"])

# How many email messages do we have from Wesley Colwell to persons of interest?
print("Total email from Wesley to POIs = ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

# Whats the value of stock options exercised by Jeffrey K Skilling?
print("Jeffrey Skilling stock options excercised = ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

# Among Lay, Skilling and Fastow, who took home the most money?
most_paid = ''
highest_payment = 0

for key in ('LAY KENNETH L', 'FASTOW ANDREW S', 'SKILLING JEFFREY K'):
	if enron_data[key]['total_payments'] > highest_payment:
		highest_payment = enron_data[key]['total_payments']
		most_paid = key

print (most_paid, highest_payment)

# How many folks have a quantified salary?
print("No. of people with quantified salary = ", len(dict((key, value) for key, value in enron_data.items() if value["salary"] != 'NaN')))

# How many with a known email address?
print("No. of people with quantified salary = ", len(dict((key, value) for key, value in enron_data.items() if value["email_address"] != 'NaN')))

# How many people have NaN for total_payments? What is the percentage of total?
no_total_payments = len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))
print("% of people with NaN as total_payments = ", float(no_total_payments)/len(enron_data) * 100)

# What percentage of POIs in the data have "NaN" for their total payments?
POIs = dict((key,value) for key, value in enron_data.items() if value['poi'] == True)
number_POIs = len(POIs)
no_total_payments = len(dict((key, value) for key, value in POIs.items() if value["total_payments"] == 'NaN'))
print("% of POIs with NaN as total_payments = ", float(no_total_payments)/number_POIs * 100)

# If 10 POIs with NaN total_payments were added, what is the new number of people? 
# What is the new number of people with NaN total_payments?
print(len(enron_data) + 10)
print(10 + len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN')))

# What is the new number of POIs?
print("New no. of POIs = ", 10 + len(POIs))

# What percentage have NaN for their total_payments?
print("% with NaN as total_payments = ", float(10)/(10 + len(POIs))*100)










