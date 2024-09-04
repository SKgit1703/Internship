from elasticsearch import Elasticsearch

# Create an instance of Elasticsearch client
es = Elasticsearch(
    [
        {
            'host': 'localhost',
            'port': 9200,
            'scheme': 'http'  # Specify the scheme
        }
    ]
)

# Check if Elasticsearch is connected
try:
    # Ping the cluster
    if es.ping():
        print("Elasticsearch is connected!")
    else:
        print("Elasticsearch is not connected.")
except Exception as e:
    print(f"Error connecting to Elasticsearch: {e}")
