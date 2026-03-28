NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_70 -Iinclude
TARGET = radix_sort
SOURCES = src/main.cu src/radix_sort.cu

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

.PHONY: all clean