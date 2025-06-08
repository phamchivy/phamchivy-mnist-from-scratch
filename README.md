# mnist-from-scratch
Code for training basic neural networks, especially the MNIST numbers dataset.
Parallelized using OpenMP!

### Get the training dataset
```
kaggle datasets download -d oddrationale/mnist-in-csv
unzip mnist-in-csv.zip -d data
```

### Pipeline Parallelism training with docker compose
```
# Build 
docker-compose -f .\docker-compose-pipeline.yml build

# Run container 
docker-compose -f .\docker-compose-pipeline.yml --env-file .\env.default up 

```

