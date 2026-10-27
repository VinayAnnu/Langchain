
import truststore
truststore.inject_into_ssl()

import arxiv

client = arxiv.Client()

# Override the HTTP endpoint used by this installed version.
client.query_url_format = "https://export.arxiv.org/api/query?{}"

search = arxiv.Search(
    id_list=["1706.03762"],
    max_results=1
)

try:
    for paper in client.results(search):
        print("Title:", paper.title)
        print("Authors:", ", ".join(a.name for a in paper.authors))
        print("Abstract:", paper.summary)
        print("URL:", paper.entry_id)

except Exception as e:
    print("Error type:", type(e).__name__)
    print("Error details:", repr(e))
