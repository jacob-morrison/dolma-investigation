import dolma
import json
from collections import Counter
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tagger = dolma.language.FastTextEnglishLanguageDocumentTagger()

documents = [
    'CANADA_S_ALL',
    'CANADA_W_ALL',
    'EAST_AFRICA_S_ALL',
    'EAST_AFRICA_W_ALL',
    'HONG_KONG_S_ALL',
    'HONG_KONG_W_ALL',
    'INDIA_S_ALL',
    'INDIA_W_ALL',
    'IRELAND_S_ALL',
    'IRELAND_W_ALL',
    'JAMAICA_S_ALL',
    'JAMAICA_W_ALL',
    'PHILIPPINES_S_ALL',
    'PHILIPPINES_W_ALL',
    'SINGAPORE_S_ALL',
    'SINGAPORE_W_ALL',
    'USA_W_ALL',
]

document_to_country = {
    'CANADA_S_ALL': 'Canada',
    'CANADA_W_ALL': 'Canada',
    'EAST_AFRICA_S_ALL': 'East Africa',
    'EAST_AFRICA_W_ALL': 'East Africa',
    'HONG_KONG_S_ALL': 'Hong Kong',
    'HONG_KONG_W_ALL': 'Hong Kong',
    'INDIA_S_ALL': 'India',
    'INDIA_W_ALL': 'India',
    'IRELAND_S_ALL': 'Ireland',
    'IRELAND_W_ALL': 'Ireland',
    'JAMAICA_S_ALL': 'Jamaica',
    'JAMAICA_W_ALL': 'Jamaica',
    'PHILIPPINES_S_ALL': 'Philippines',
    'PHILIPPINES_W_ALL': 'Philippines',
    'SINGAPORE_S_ALL': 'Singapore',
    'SINGAPORE_W_ALL': 'Singapore',
    'USA_W_ALL': 'United States',
}

total_counts = Counter()
total_counts_by_country = Counter()
unfiltered_counts = Counter()
threshold_counts = {}
threshold_counts_by_country = {}
step = 0.001
thresholds = np.arange(0, 1. + step, step)

for document in documents:
    threshold_counts[document] = {}
    threshold_counts_by_country[document_to_country[document]] = {}
    for threshold in thresholds:
        threshold_counts[document][threshold] = 0
        threshold_counts_by_country[document_to_country[document]][threshold] = 0

for document in documents:
    with open('./data/' + document + '.jsonl') as f_in:
        with open('./tagged_documents/' + document + '.jsonl', 'w') as f_out:
            i = 0
            for line in f_in.readlines():
                i += 1
                data = json.loads(line)
                data['text'] = str(data['text'])
                text = str(data['text'])
                if i == 130 and document == 'USA_W_ALL':
                    text = text_str
                fake_document = dolma.core.data_types.Document(
                    source='my brain',
                    version=0.1,
                    id='fake-document-0.1',
                    text=str(text)
                )
                result = tagger.predict(fake_document)
                data['english_score'] = result.spans[0].score
                data['non_english_score'] = result.spans[1].score
                total_counts[document] += 1
                total_counts_by_country[document_to_country[document]] += 1
                for threshold in thresholds:
                    if data['english_score'] > threshold:
                        threshold_counts[document][threshold] += 1
                        threshold_counts_by_country[document_to_country[document]][threshold] += 1
                    elif threshold == 0.50:
                        print(str(data['text']))
                f_out.write(str(data) + '\n')

with open('./tagged_documents/summary_statistics.txt', 'w') as stats:
    stats.write('Summary statistics for document-level FastText tagging: \n\n')
    for document in documents:
        stats.write('Total counts for ' + document + ': ' + str(total_counts[document]) + '\n')
        stats.write('Unfiltered counts for ' + document + ': ' + str(threshold_counts[document][0.50]) + '\n')
        stats.write('Filtered percentage for ' + document + ': ' + str(1.0 - (threshold_counts[document][0.50] / total_counts[document])) + '\n\n')

filter_percentages = []
filter_percentages_by_country = []
seen = set()
for document in documents:
    # filter_percentages[document] = {}
    for threshold in thresholds:
        percentage = 1.0 - (threshold_counts[document][threshold] / total_counts[document])
        percentage_by_country = 1.0 - (threshold_counts_by_country[document_to_country[document]][threshold] / total_counts_by_country[document_to_country[document]])
        filter_percentages.append({
            'Document': document,
            'Country': document.split('_')[0],
            'Threshold': threshold,
            'Filter percentage': percentage,
        })
        if (threshold, document_to_country[document]) not in seen:
            seen.add((threshold, document_to_country[document]))
            filter_percentages_by_country.append({
                'Document': document,
                'Country': document_to_country[document],
                'Threshold': threshold,
                'Filter percentage': percentage_by_country,
            })

filter_df = pd.DataFrame(filter_percentages)
filter_by_country_df = pd.DataFrame(filter_percentages_by_country)
print(filter_df)
print(filter_by_country_df)

fig, ax = plt.subplots()

# filter_by = 'COUNTRY'
filter_by = 'REGION'

# Set the style and context
sns.set_style("whitegrid")
sns.set_context("talk")

# Increase the label font sizes
plt.xlabel('Threshold', fontsize=18)  # Adjust the font size as needed
plt.ylabel('Filter Percentage', fontsize=18)

plt.xticks(fontsize=14)  # Adjust the font size as needed for x-axis
plt.yticks(fontsize=14)  # Adjust the font size as needed for y-axis

if filter_by == 'COUNTRY':
    sns.lineplot(data=filter_by_country_df, x = 'Threshold', y = 'Filter percentage', hue = 'Document')
elif filter_by == 'REGION':
    sns.lineplot(data=filter_df, x = 'Threshold', y = 'Filter percentage', hue = 'Document')
else:
    print('invalid')
    quit()
ax.set_xlim(0.8, 1.0)
plt.show()

