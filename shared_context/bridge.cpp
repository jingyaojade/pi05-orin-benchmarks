// Embed the exact tested llama.cpp implementation; no Qwen inference rewrite.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cupti.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

int llama_server(int argc, char ** argv);
void llama_server_terminate();

static thread_local int role = 0; // 1: VLA request thread; 2: embedded llama loop
static std::atomic<bool> auditing{false};
static CUpti_SubscriberHandle subscriber;
static std::mutex audit_mutex;
struct Event { uint64_t time; uintptr_t context; uint32_t uid; long tid; int role; std::string api; };
static std::vector<Event> events;
static uint64_t discarded = 0;
static std::string result;

static void CUPTIAPI callback(void *, CUpti_CallbackDomain domain, CUpti_CallbackId, const void * raw) {
    if (!auditing.load() || (domain != CUPTI_CB_DOMAIN_RUNTIME_API && domain != CUPTI_CB_DOMAIN_DRIVER_API)) return;
    const auto * data = static_cast<const CUpti_CallbackData *>(raw);
    if (data->callbackSite != CUPTI_API_ENTER || !data->functionName || !std::strstr(data->functionName, "Launch")) return;
    if (!data->context) return;
    const uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(audit_mutex);
    if (events.size() >= 20000) { ++discarded; return; }
    events.push_back({now, reinterpret_cast<uintptr_t>(data->context), data->contextUid,
                      syscall(SYS_gettid), role, data->functionName});
}

extern "C" {
int sc_llama_run(int argc, char ** argv) {
    role = 2;
    const auto status = cudaSetDevice(0);
    if (status != cudaSuccess) return -int(status);
    return llama_server(argc, argv);
}
void sc_llama_stop() { llama_server_terminate(); }
void sc_set_role(int value) { role = value; }
int sc_set_context(uintptr_t value) { return int(cuCtxSetCurrent(reinterpret_cast<CUcontext>(value))); }
uintptr_t sc_current_context() {
    CUcontext ctx = nullptr;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS) return 0;
    return reinterpret_cast<uintptr_t>(ctx);
}
int sc_audit_start() {
    if (auditing.load()) return -1;
    { std::lock_guard<std::mutex> lock(audit_mutex); events.clear(); discarded = 0; }
    auto status = cuptiSubscribe(&subscriber, callback, nullptr);
    if (status != CUPTI_SUCCESS) return int(status);
    status = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    if (status != CUPTI_SUCCESS) { cuptiUnsubscribe(subscriber); return int(status); }
    status = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    if (status != CUPTI_SUCCESS) { cuptiUnsubscribe(subscriber); return int(status); }
    auditing.store(true);
    return 0;
}
const char * sc_audit_stop() {
    auditing.store(false);
    cuptiUnsubscribe(subscriber);
    std::lock_guard<std::mutex> lock(audit_mutex);
    std::ostringstream out;
    out << "{\"discarded\":" << discarded << ",\"events\":[";
    bool first = true;
    for (const auto & e : events) {
        if (!first) out << ',';
        first = false;
        out << "{\"monotonic_ns\":" << e.time << ",\"context\":" << e.context
            << ",\"context_uid\":" << e.uid << ",\"tid\":" << e.tid
            << ",\"role\":" << e.role << ",\"api\":\"" << e.api << "\"}";
    }
    out << "]}";
    result = out.str();
    return result.c_str();
}
int sc_profiler_start() { return int(cudaProfilerStart()); }
int sc_profiler_stop() { return int(cudaProfilerStop()); }
}
