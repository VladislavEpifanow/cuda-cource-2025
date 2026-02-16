NVCC = nvcc
CFLAGS = -O2 -arch=sm_70 -I../include -std=c++17

TARGET = program

all: $(TARGET)

$(TARGET): src/main.cu src/gpu_radix_sort.cu
	$(NVCC) $(CFLAGS) -o $(TARGET) $^

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
