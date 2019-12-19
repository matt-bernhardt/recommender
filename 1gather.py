import json

# This script takes a list of JSON files and gathers them together into one
# file.

print('Beginning...')

files = [
  '2019-03.json',
  '2019-05.json',
  '2019-07.json',
  '2019-08.json',
  '2019-09.json',
  '2019-10.json',
  '2019-11.json',
  '2019-12.json'
]
# PLEASE NOTE: Some records were unable to be imported. They were removed from
# these files and placed in data/unused.json.

combined = []

for file in files:
  print(str(file))
  with open('data/' + file, encoding="utf8") as myfile:
    data = myfile.read()

  obj = json.loads(data)
  print(str(len(obj)))

  for record in obj:
    print('=================================================================')
    print(str(len(combined)) + ' records and counting')
    print(str(type(record)))
    print(str(record["identifier"]))
    print(str(record["title"]))
    print('=================================================================')
    print(str(record))
    combined.append(record)

with open('data/combined.json', 'w') as outfile:
  json.dump(combined, outfile)
