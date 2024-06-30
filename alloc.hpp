#ifndef alloc_hpp
#define alloc_hpp

#define DEVICE_ALLOCATE(name, count, type) \
  type *name; cudaMalloc(&name, count * sizeof(type))

#define DEVICE_FREE(name) \
  cudaFree(name)

#define HOST_ALLOCATE(name, count, type) \
  type *name = new type[count]

#define HOST_FREE(name) \
  delete name

#endif
