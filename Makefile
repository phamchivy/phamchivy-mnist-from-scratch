# Compiler and flags
CC = gcc
CFLAGS = -Wall -fopenmp
LDFLAGS = -lm

# Sources and headers
C_SOURCES = $(wildcard matrix/*.c neural/*.c util/*.c socket/*.c)
HEADERS = $(wildcard matrix/*.h neural/*.h util/*.h *.h socket/*.h)

# Object files
OBJ = $(C_SOURCES:.c=.o)

# Executable
TARGET = app
PREDICT = predict

# Default target
all: $(TARGET) $(PREDICT)

# Build the app
$(TARGET): train.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Build predict
$(PREDICT): predict.c $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Object file rule
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean target
clean:
	rm -f matrix/*.o neural/*.o util/*.o socket/*.o *.o $(TARGET) $(PREDICT) 