from __future__ import absolute_import
import json

# This script takes the combined list of records, removes the subject element,
# and saves the resulting pruned file.
#
# The intent right now is to use the subject headings as a sanity check for
# our initial results - to see how our recommendations would compare to
# browsing by subject heading.

print('Beginning...')

file = 'combined.json'

result = []
summary = {}

with open('data/combined.json', encoding="utf8") as myfile:
  records = json.load(myfile)

for record in records:
  print('===================================================================')
  for item in record:
    # Loop over every item in the record
    if item in summary.keys():
      summary[item] += 1
    else:
      summary[item] = 1
    print(str(item))
  record.pop('subjects',None)
  result.append(record)

print(str(summary))

with open('data/pruned.json', 'w') as outfile:
  json.dump(result, outfile)

with open('data/summary.json', 'w') as outfile:
  json.dump(summary, outfile)
