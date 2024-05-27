# Makefile

# Compiler settings
CC=gcc

# Compiler flags
CFLAGS=-O3
LDFLAGS=-lm -lopenblas

# Target binary
TARGET=final

# Source file
SOURCE=final.c

# Default rule
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean rule
clean:
	rm -f $(TARGET)

.PHONY: all clean
