# Hybrid Parallelism - Pipeline + Data Parallelism

## Architecture Overview

```
Pipeline Group 1 (Data 0-29999):          Pipeline Group 2 (Data 30000-59999):
Worker1_Stage1 ──activation──> Worker1_Stage2    Worker2_Stage1 ──activation──> Worker2_Stage2
(784→300)                      (300→10)          (784→300)                      (300→10)
    │                              │                  │                              │
    └──────────Hidden weights──────┼──────────────────┴────────Output weights────────┘
                                   │
                            Parameter Server
                      (Hidden Center + Output Center)
```

## Key Features

### **Pipeline Parallelism**
- **Stage 1**: Input (784) → Hidden (300) using Sigmoid activation
- **Stage 2**: Hidden (300) → Output (10) using Sigmoid + Softmax
- **Asynchronous Communication**: Stage 1 can sync independently with Parameter Server

### **Data Parallelism**  
- **Group 1**: Processes images 0-29999
- **Group 2**: Processes images 30000-59999
- Each group runs its own 2-stage pipeline

### **Elastic Averaging**
- **Separate Centers**: Parameter Server maintains separate elastic centers for Hidden and Output weights
- **Independent Sync**: Stage 1 workers sync every 1000 images without waiting for Stage 2
- **EASGD Parameters**: α=0.01 (elasticity), β=0.001 (server learning rate)

## How to Run

### 1. Build the Hybrid System
```bash
make hybrid
```

### 2. Start with Docker Compose
```bash
docker-compose -f docker-compose-hybrid.yml up --build
```

### 3. Monitor Logs
```bash
docker-compose -f docker-compose-hybrid.yml logs -f
```

## Network Architecture

| Service | IP Address | Ports | Role |
|---------|------------|-------|------|
| parameter_server | 172.32.0.10 | 12345 | Central coordination |
| worker1_stage1 | 172.32.0.11 | - | Group 1 pipeline input |
| worker1_stage2 | 172.32.0.12 | 13001 | Group 1 pipeline output |
| worker2_stage1 | 172.32.0.13 | - | Group 2 pipeline input |
| worker2_stage2 | 172.32.0.14 | 13002 | Group 2 pipeline output |

## Communication Flow

### 1. Forward Pass
```
Stage1 → Matrix(300x1) activation → Stage2
```

### 2. Backward Pass  
```
Stage2 → Matrix(300x1) gradient → Stage1
```

### 3. Weight Synchronization
```
Stage1 → Hidden weights (300x784) → Parameter Server
Stage2 → Output weights (10x300) → Parameter Server
```

## Performance Benefits

1. **Reduced Memory**: Each worker only holds partial model
2. **Parallel Training**: 4 workers train simultaneously  
3. **Asynchronous Sync**: No blocking between pipeline stages
4. **Elastic Averaging**: Prevents divergence in distributed training

## File Structure

```
apps/
├── parameter_server.c     # Updated for separate weight handling
├── worker_stage1.c        # Pipeline stage 1 worker
└── worker_stage2.c        # Pipeline stage 2 worker

neural/
├── pipeline_utils.h/c     # Pipeline-specific functions
└── nn.h/c                 # Updated with separate elastic centers

socket/
└── pipeline_socket.h/c    # Pipeline communication functions

dockerfiles/
├── Dockerfile.worker_stage1
└── Dockerfile.worker_stage2
```

## Expected Behavior

- **Stage 1 Workers**: Process input data, send activations, sync hidden weights independently
- **Stage 2 Workers**: Receive activations, compute final output, sync output weights  
- **Parameter Server**: Maintain separate elastic centers, handle async weight updates
- **Training Progress**: Each group processes 30k images with regular accuracy reporting

## Troubleshooting

1. **Connection Issues**: Check if ports 13001, 13002 are available
2. **Memory Issues**: Monitor Docker container memory usage
3. **Sync Issues**: Check parameter server logs for weight type mismatches
4. **Pipeline Stalls**: Verify Stage 2 workers are accepting connections

## Cleanup

```bash
docker-compose -f docker-compose-hybrid.yml down
make clean_all
```