import dolma

fake_document_text = 'this is my full document text that I\'m testing things out with'
fake_document = dolma.core.data_types.Document(
    'my brain',
    0.1,
    'fake-document-0.1',
    fake_document_text
)
fake_text_slice = dolma.core.data_types.TextSlice(fake_document.text, 0, 16)

print('hello')
tagger = dolma.language.FastTextEnglishLanguageParagraphWithDocScoreTagger()
print(tagger.predict(doc = fake_document))
