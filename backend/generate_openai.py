from machine_learning.embedder import generate_embedding

generate_embedding('openai', 'datasets/dataset_validation/raw/all_data.csv', batch_size=50)