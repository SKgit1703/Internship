from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser

# Define the schema for the index
schema = Schema(title=TEXT(stored=True), content=TEXT(stored=True))

# Create an index directory
import os
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

# Create an index in the directory
ix = create_in("indexdir", schema)

# Write a document to the index
writer = ix.writer()
writer.add_document(title="First Document", content="This is the content of the first document.")
writer.commit()

# Search the index
searcher = ix.searcher()
query = QueryParser("content", ix.schema).parse("first")
results = searcher.search(query)

# Display the results
print("Found", len(results), "result(s).")
for result in results:
    print("Title:", result["title"])
    print("Content:", result["content"])

# Close the searcher
searcher.close()
