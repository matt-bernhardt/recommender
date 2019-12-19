from __future__ import absolute_import
import json

# This script takes a list of JSON files and gathers them together into one
# file.

print('Hello, world!')

fields=[
  'identifier',
  'source',
  'source_link',
  'title',
  'languages',
  'publication_date',
  'physical_description',
  'notes',
  'holdings',
  'summary',
  'links',
  'contributors',
  'citation'
]

result = []
summary = {}

with open('data/pruned.json', encoding="utf8") as myfile:
  records = json.load(myfile)

for record in records:
  print('===================================================================')
  print(str(record))
  new_record = {}
  for field in fields:
    print(str(field))
    #print(str(record[field]))
    value = 0
    field_type = ''
    if field in record:
      print(str(type(record[field])))
      if isinstance(record[field], str):
        value = record[field]
        field_type = 'string'
      elif isinstance(record[field], list):
        value = str(record[field])
        field_type = 'list'
      else:
        value = 'unknown'
        field_type = 'unknown'
    else:
      value = ''
      field_type = 'null'
    new_record[field] = value
    print(str(field) + ': (' + str(field_type) + ') ' + str(value))
    print('')
    print('')
  result.append(new_record)

with open('data/rect-objects.json', 'w') as outfile:
  json.dump(result, outfile)
