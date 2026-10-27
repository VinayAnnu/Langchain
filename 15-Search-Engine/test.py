import truststore
truststore.inject_into_ssl()

from ddgs import DDGS

with DDGS(timeout=20) as ddgs:
    results = ddgs.text(
        "LangChain documentation",
        max_results=3,
        backend="bing",
    )

    for result in results:
        print(result)