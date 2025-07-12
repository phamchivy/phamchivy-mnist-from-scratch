# Compiler and flags
CC = gcc
CFLAGS = -Wall -fopenmp
LDFLAGS = -lm

# Sources and headers
C_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c socket/*.c)
PIPELINE_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c socket/*.c pipeline/*.c)
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h *.h socket/*.h pipeline/*.h)

# Object files
OBJ = $(C_SOURCES:.c=.o)
PIPELINE_OBJ = $(PIPELINE_SOURCES:.c=.o)

# Executable
TARGET = app
PIPELINE_TARGET = app_pipeline
PREDICT = predict

# Default target
all: $(TARGET) $(PIPELINE_TARGET)

# Build the app
$(TARGET): train.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build the pipeline app
$(PIPELINE_TARGET): train_pipeline.c $(PIPELINE_OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) -lpthread

# Build predict
$(PREDICT): predict.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build single training
single_train: train_single.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Object file rule
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f matrix/*.o neural/*.o util/*.o socket/*.o pipeline/*.o *.o $(TARGET) $(PIPELINE_TARGET) $(PREDICT) single_train
