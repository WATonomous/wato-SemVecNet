# TensorRT Memory Management: Shared Pointers vs Raw Pointers

## TL;DR
**For `main.cpp`:** Raw pointers with manual `destroy()` are fine and simpler.  
**For classes/long-lived objects:** Use `std::shared_ptr` with custom deleters.

---

## Option 1: Raw Pointers (Simple, for `main.cpp`)

```cpp
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cerr << "[TensorRT] " << msg << std::endl;
  }
} gLogger;

int main() {
  // Load .plan file (omitted for brevity)
  std::vector<uint8_t> data = loadFile("model.plan");
  
  // Create runtime (raw pointer)
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
  if (!runtime) return 1;
  
  // Deserialize engine (raw pointer)
  nvinfer1::ICudaEngine* engine = 
      runtime->deserializeCudaEngine(data.data(), data.size());
  if (!engine) {
    runtime->destroy();  // Clean up runtime
    return 1;
  }
  
  // Create execution context (raw pointer)
  nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  if (!context) {
    engine->destroy();
    runtime->destroy();
    return 1;
  }
  
  // Use context for inference...
  // ...
  
  // Manual cleanup (MUST do this in reverse order)
  context->destroy();
  engine->destroy();
  runtime->destroy();
  
  return 0;
}
```

**Pros:**
- Simple and clear
- No shared_ptr overhead
- Fine for simple programs

**Cons:**
- Must remember to call `destroy()` in reverse order
- Easy to leak if you forget or have early returns
- Need to track ownership manually

---

## Option 2: Shared Pointers (Safer, for classes)

```cpp
#include <NvInfer.h>
#include <memory>

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cerr << "[TensorRT] " << msg << std::endl;
  }
} gLogger;

// Custom deleters for TensorRT objects
template<typename T>
struct TensorRTDeleter {
  void operator()(T* obj) {
    if (obj) obj->destroy();
  }
};

class TensorRTEngine {
 public:
  bool loadModel(const std::string& planPath) {
    auto data = loadFile(planPath);
    if (data.empty()) return false;
    
    // Create runtime with custom deleter
    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) return false;
    
    // Create engine with custom deleter
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) return false;
    
    // Create context with custom deleter
    context_.reset(engine_->createExecutionContext());
    return context_ != nullptr;
  }
  
  // Automatic cleanup when object is destroyed!
  ~TensorRTEngine() = default;
  
  nvinfer1::IExecutionContext* getContext() { return context_.get(); }
  
 private:
  std::unique_ptr<nvinfer1::IRuntime, TensorRTDeleter<nvinfer1::IRuntime>> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TensorRTDeleter<nvinfer1::ICudaEngine>> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TensorRTDeleter<nvinfer1::IExecutionContext>> context_;
};

int main() {
  TensorRTEngine engine;
  if (!engine.loadModel("model.plan")) {
    std::cerr << "Failed to load model\n";
    return 1;
  }
  
  // Use engine...
  
  // Automatic cleanup when engine goes out of scope!
  return 0;
}
```

**Pros:**
- Automatic cleanup (RAII)
- Exception-safe
- Can't forget to destroy
- Good for classes/long-lived objects

**Cons:**
- More boilerplate
- Slightly more complex

---

## BEVFusion's Approach (Hybrid)

BEVFusion uses shared_ptr but still calls `destroy()` explicitly:

```cpp
class __native_engine_context {
 public:
  std::shared_ptr<nvinfer1::IRuntime> runtime_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  
  bool construct(const void *pdata, size_t size) {
    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(gLogger_),
        destroy_pointer<nvinfer1::IRuntime>  // Custom deleter
    );
    
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(pdata, size),
        destroy_pointer<nvinfer1::ICudaEngine>
    );
    
    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        destroy_pointer<nvinfer1::IExecutionContext>
    );
    
    return context_ != nullptr;
  }
};

template <typename T>
static void destroy_pointer(T *ptr) {
  if (ptr) ptr->destroy();  // Calls TensorRT's destroy()
}
```

---

## Recommendation

**For your SemVecNet project:**

1. **In `main.cpp` (simple test programs):** Use **raw pointers** - it's clearer
   
   ```cpp
   nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(...);
   // ... use it ...
   engine->destroy();
   ```

2. **In classes (production inference code):** Use **unique_ptr with custom deleter**
   
   ```cpp
   std::unique_ptr<nvinfer1::ICudaEngine, 
                   TensorRTDeleter<nvinfer1::ICudaEngine>> engine_;
   ```

3. **Why not shared_ptr?** You rarely need shared ownership for TensorRT objects.
   Use `unique_ptr` unless you're explicitly sharing the engine across threads.

---

## Example: Your test_plan_loader.cpp (Raw Pointers)

```cpp
int main(int argc, char** argv) {
  nvinfer1::IRuntime* runtime = nullptr;
  nvinfer1::ICudaEngine* engine = nullptr;
  nvinfer1::IExecutionContext* context = nullptr;
  
  // Load and deserialize
  runtime = nvinfer1::createInferRuntime(gLogger);
  if (!runtime) goto cleanup;
  
  engine = runtime->deserializeCudaEngine(data.data(), data.size());
  if (!engine) goto cleanup;
  
  context = engine->createExecutionContext();
  if (!context) goto cleanup;
  
  // Use context...
  
cleanup:
  if (context) context->destroy();
  if (engine) engine->destroy();
  if (runtime) runtime->destroy();
  
  return (context != nullptr) ? 0 : 1;
}
```

This is **perfectly fine** for simple programs. No need for smart pointers!
