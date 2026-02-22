/*
 * DeCloud GPU Proxy — Protocol header unit tests
 *
 * Verifies struct sizes, packing, magic values, and helper functions
 * from gpu_proxy_proto.h.  No CUDA dependency — runs on any machine.
 *
 * Build:  gcc -Wall -Wextra -o test_proto test_proto.c
 * Run:    ./test_proto
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../proto/gpu_proxy_proto.h"

/* ================================================================
 * Helpers
 * ================================================================ */

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST(name) \
    static void test_##name(void); \
    static void run_test_##name(void) { \
        g_tests_run++; \
        printf("  %-50s ", #name); \
        test_##name(); \
        g_tests_passed++; \
        printf("PASS\n"); \
    } \
    static void test_##name(void)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            printf("FAIL\n    %s:%d: %s (%lu) != %s (%lu)\n", \
                   __FILE__, __LINE__, #a, (unsigned long)(a), \
                   #b, (unsigned long)(b)); \
            exit(1); \
        } \
    } while (0)

#define ASSERT_TRUE(x) \
    do { \
        if (!(x)) { \
            printf("FAIL\n    %s:%d: %s is false\n", \
                   __FILE__, __LINE__, #x); \
            exit(1); \
        } \
    } while (0)

/* ================================================================
 * Tests — magic and constants
 * ================================================================ */

TEST(magic_value)
{
    ASSERT_EQ(GPU_PROXY_MAGIC, 0x44435544);
}

TEST(version_is_one)
{
    ASSERT_EQ(GPU_PROXY_VERSION, 1);
}

TEST(default_port)
{
    ASSERT_EQ(GPU_PROXY_PORT, 9999);
}

TEST(max_payload_is_64mb)
{
    ASSERT_EQ(GPU_PROXY_MAX_PAYLOAD, 64 * 1024 * 1024);
}

TEST(max_kernel_params)
{
    ASSERT_EQ(GPU_MAX_KERNEL_PARAMS, 64);
}

TEST(default_kernel_timeout)
{
    ASSERT_EQ(GPU_PROXY_DEFAULT_KERNEL_TIMEOUT_US, 30ULL * 1000000ULL);
}

/* ================================================================
 * Tests — struct sizes (packed)
 * ================================================================ */

TEST(header_is_16_bytes)
{
    ASSERT_EQ(sizeof(GpuProxyHeader), 16);
}

TEST(hello_request_is_12_bytes)
{
    /* shim_version(4) + pid(4) + conn_mode(4) = 12 */
    ASSERT_EQ(sizeof(GpuHelloRequest), 12);
}

TEST(hello_response_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuHelloResponse), 8);
}

TEST(get_device_count_response_is_4_bytes)
{
    ASSERT_EQ(sizeof(GpuGetDeviceCountResponse), 4);
}

TEST(malloc_request_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuMallocRequest), 8);
}

TEST(malloc_response_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuMallocResponse), 8);
}

TEST(free_request_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuFreeRequest), 8);
}

TEST(memcpy_request_is_28_bytes)
{
    ASSERT_EQ(sizeof(GpuMemcpyRequest), 28);
}

TEST(memset_request_is_20_bytes)
{
    ASSERT_EQ(sizeof(GpuMemsetRequest), 20);
}

TEST(register_module_request_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuRegisterModuleRequest), 8);
}

TEST(register_module_response_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuRegisterModuleResponse), 8);
}

TEST(register_function_request_is_20_bytes)
{
    ASSERT_EQ(sizeof(GpuRegisterFunctionRequest), 20);
}

TEST(register_function_response_size)
{
    /* num_params(4) + param_sizes[64]*4 = 4 + 256 = 260 */
    ASSERT_EQ(sizeof(GpuRegisterFunctionResponse), 260);
}

TEST(launch_kernel_request_is_56_bytes)
{
    ASSERT_EQ(sizeof(GpuLaunchKernelRequest), 56);
}

TEST(stream_create_request_is_4_bytes)
{
    ASSERT_EQ(sizeof(GpuStreamCreateRequest), 4);
}

TEST(stream_create_response_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuStreamCreateResponse), 8);
}

TEST(event_create_request_is_4_bytes)
{
    ASSERT_EQ(sizeof(GpuEventCreateRequest), 4);
}

TEST(event_elapsed_time_response_is_4_bytes)
{
    ASSERT_EQ(sizeof(GpuEventElapsedTimeResponse), 4);
}

TEST(set_memory_quota_request_is_8_bytes)
{
    ASSERT_EQ(sizeof(GpuSetMemoryQuotaRequest), 8);
}

TEST(usage_stats_response_size)
{
    /* 8+8+8+8+4+4+8+8 = 56 bytes */
    ASSERT_EQ(sizeof(GpuUsageStatsResponse), 56);
}

TEST(report_usage_stats_size)
{
    /* 8+8+8+4+4+4+4+4+4+8 = 56 bytes */
    ASSERT_EQ(sizeof(GpuReportUsageStats), 56);
}

TEST(connection_modes)
{
    ASSERT_EQ(GPU_CONN_PROXY, 0);
    ASSERT_EQ(GPU_CONN_METER, 1);
}

TEST(agent_report_interval)
{
    ASSERT_EQ(GPU_AGENT_REPORT_INTERVAL_SEC, 5);
}

TEST(report_cmd_id)
{
    ASSERT_EQ(GPU_CMD_REPORT_USAGE_STATS, 0x62);
}

/* ================================================================
 * Tests — header field offsets (verify packing)
 * ================================================================ */

TEST(header_field_offsets)
{
    GpuProxyHeader hdr;
    char *base = (char *)&hdr;

    ASSERT_EQ((char *)&hdr.magic       - base, 0);
    ASSERT_EQ((char *)&hdr.version     - base, 4);
    ASSERT_EQ((char *)&hdr.cmd         - base, 5);
    ASSERT_EQ((char *)&hdr.flags       - base, 6);
    ASSERT_EQ((char *)&hdr.payload_len - base, 8);
    ASSERT_EQ((char *)&hdr.status      - base, 12);
}

/* ================================================================
 * Tests — command ID ranges
 * ================================================================ */

TEST(command_id_ranges)
{
    /* Device mgmt: 0x01-0x03 */
    ASSERT_EQ(GPU_CMD_GET_DEVICE_COUNT, 0x01);
    ASSERT_EQ(GPU_CMD_SET_DEVICE, 0x03);

    /* Memory: 0x10-0x13 */
    ASSERT_EQ(GPU_CMD_MALLOC, 0x10);
    ASSERT_EQ(GPU_CMD_MEMSET, 0x13);

    /* Execution: 0x20-0x21 */
    ASSERT_EQ(GPU_CMD_LAUNCH_KERNEL, 0x20);
    ASSERT_EQ(GPU_CMD_DEVICE_SYNCHRONIZE, 0x21);

    /* Streams: 0x30-0x32 */
    ASSERT_EQ(GPU_CMD_STREAM_CREATE, 0x30);
    ASSERT_EQ(GPU_CMD_STREAM_SYNCHRONIZE, 0x32);

    /* Events: 0x40-0x44 */
    ASSERT_EQ(GPU_CMD_EVENT_CREATE, 0x40);
    ASSERT_EQ(GPU_CMD_EVENT_ELAPSED_TIME, 0x44);

    /* Module/function: 0x50-0x53 */
    ASSERT_EQ(GPU_CMD_REGISTER_MODULE, 0x50);
    ASSERT_EQ(GPU_CMD_REGISTER_VAR, 0x53);

    /* Resource mgmt: 0x60-0x62 */
    ASSERT_EQ(GPU_CMD_SET_MEMORY_QUOTA, 0x60);
    ASSERT_EQ(GPU_CMD_GET_USAGE_STATS, 0x61);
    ASSERT_EQ(GPU_CMD_REPORT_USAGE_STATS, 0x62);

    /* Lifecycle: 0xF0-0xF1 */
    ASSERT_EQ(GPU_CMD_HELLO, 0xF0);
    ASSERT_EQ(GPU_CMD_GOODBYE, 0xF1);
}

/* ================================================================
 * Tests — helper functions
 * ================================================================ */

TEST(memcpy_h2d_payload_len)
{
    uint64_t data_size = 1024;
    uint32_t expected = (uint32_t)(sizeof(GpuMemcpyRequest) + data_size);
    ASSERT_EQ(gpu_memcpy_h2d_payload_len(data_size), expected);
}

TEST(memcpy_h2d_payload_len_zero)
{
    ASSERT_EQ(gpu_memcpy_h2d_payload_len(0), (uint32_t)sizeof(GpuMemcpyRequest));
}

/* ================================================================
 * Tests — wire format round-trip (serialize/deserialize)
 * ================================================================ */

TEST(header_roundtrip)
{
    GpuProxyHeader hdr = {
        .magic       = GPU_PROXY_MAGIC,
        .version     = GPU_PROXY_VERSION,
        .cmd         = GPU_CMD_MALLOC,
        .flags       = 0,
        .payload_len = 42,
        .status      = 0,
    };

    /* Serialize to bytes */
    char buf[sizeof(GpuProxyHeader)];
    memcpy(buf, &hdr, sizeof(hdr));

    /* Deserialize */
    GpuProxyHeader out;
    memcpy(&out, buf, sizeof(out));

    ASSERT_EQ(out.magic, GPU_PROXY_MAGIC);
    ASSERT_EQ(out.version, GPU_PROXY_VERSION);
    ASSERT_EQ(out.cmd, GPU_CMD_MALLOC);
    ASSERT_EQ(out.flags, 0);
    ASSERT_EQ(out.payload_len, 42);
    ASSERT_EQ(out.status, 0);
}

TEST(launch_kernel_roundtrip)
{
    GpuLaunchKernelRequest req = {
        .host_func_ptr   = 0xDEADBEEF,
        .grid_dim_x      = 128,
        .grid_dim_y      = 1,
        .grid_dim_z      = 1,
        .block_dim_x     = 256,
        .block_dim_y     = 1,
        .block_dim_z     = 1,
        .shared_mem_bytes = 4096,
        .stream_handle   = 0,
        .num_params      = 3,
        .args_total_size = 24,
    };

    char buf[sizeof(req)];
    memcpy(buf, &req, sizeof(req));

    GpuLaunchKernelRequest out;
    memcpy(&out, buf, sizeof(out));

    ASSERT_EQ(out.host_func_ptr, 0xDEADBEEF);
    ASSERT_EQ(out.grid_dim_x, 128);
    ASSERT_EQ(out.block_dim_x, 256);
    ASSERT_EQ(out.shared_mem_bytes, 4096);
    ASSERT_EQ(out.num_params, 3);
    ASSERT_EQ(out.args_total_size, 24);
}

TEST(usage_stats_roundtrip)
{
    GpuUsageStatsResponse stats = {
        .memory_allocated = 1024 * 1024 * 512,
        .memory_quota     = 1024ULL * 1024 * 1024 * 2,
        .peak_memory      = 1024ULL * 1024 * 768,
        .total_alloc_bytes = 1024ULL * 1024 * 1024,
        .kernel_launches  = 42,
        .kernel_timeouts  = 1,
        .kernel_time_us   = 123456789,
        .connect_time_us  = 987654321,
    };

    char buf[sizeof(stats)];
    memcpy(buf, &stats, sizeof(stats));

    GpuUsageStatsResponse out;
    memcpy(&out, buf, sizeof(out));

    ASSERT_EQ(out.memory_allocated, 1024 * 1024 * 512);
    ASSERT_EQ(out.memory_quota, 1024ULL * 1024 * 1024 * 2);
    ASSERT_EQ(out.kernel_launches, 42);
    ASSERT_EQ(out.kernel_timeouts, 1);
    ASSERT_EQ(out.kernel_time_us, 123456789);
}

TEST(report_usage_stats_roundtrip)
{
    GpuReportUsageStats stats = {
        .memory_allocated = 2ULL * 1024 * 1024 * 1024,
        .memory_total     = 24ULL * 1024 * 1024 * 1024,
        .peak_memory      = 3ULL * 1024 * 1024 * 1024,
        .gpu_utilization  = 85,
        .mem_utilization  = 42,
        .temperature_c    = 72,
        .fan_speed_pct    = 55,
        .power_usage_mw   = 250000,
        .power_limit_mw   = 350000,
        .uptime_us        = 3600000000ULL,
    };

    char buf[sizeof(stats)];
    memcpy(buf, &stats, sizeof(stats));

    GpuReportUsageStats out;
    memcpy(&out, buf, sizeof(out));

    ASSERT_EQ(out.memory_allocated, 2ULL * 1024 * 1024 * 1024);
    ASSERT_EQ(out.memory_total, 24ULL * 1024 * 1024 * 1024);
    ASSERT_EQ(out.gpu_utilization, 85);
    ASSERT_EQ(out.temperature_c, 72);
    ASSERT_EQ(out.power_usage_mw, 250000);
    ASSERT_EQ(out.power_limit_mw, 350000);
    ASSERT_EQ(out.uptime_us, 3600000000ULL);
}

TEST(hello_with_conn_mode_roundtrip)
{
    GpuHelloRequest req = {
        .shim_version = 1,
        .pid          = 12345,
        .conn_mode    = GPU_CONN_METER,
    };

    char buf[sizeof(req)];
    memcpy(buf, &req, sizeof(req));

    GpuHelloRequest out;
    memcpy(&out, buf, sizeof(out));

    ASSERT_EQ(out.shim_version, 1);
    ASSERT_EQ(out.pid, 12345);
    ASSERT_EQ(out.conn_mode, GPU_CONN_METER);
}

/* ================================================================
 * Tests — memcpy kind mapping
 * ================================================================ */

TEST(memcpy_kinds)
{
    ASSERT_EQ(GPU_MEMCPY_HOST_TO_HOST, 0);
    ASSERT_EQ(GPU_MEMCPY_HOST_TO_DEVICE, 1);
    ASSERT_EQ(GPU_MEMCPY_DEVICE_TO_HOST, 2);
    ASSERT_EQ(GPU_MEMCPY_DEVICE_TO_DEVICE, 3);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void)
{
    printf("GPU Proxy Protocol Tests\n");
    printf("========================\n");

    /* Constants */
    run_test_magic_value();
    run_test_version_is_one();
    run_test_default_port();
    run_test_max_payload_is_64mb();
    run_test_max_kernel_params();
    run_test_default_kernel_timeout();

    /* Struct sizes */
    run_test_header_is_16_bytes();
    run_test_hello_request_is_12_bytes();
    run_test_hello_response_is_8_bytes();
    run_test_get_device_count_response_is_4_bytes();
    run_test_malloc_request_is_8_bytes();
    run_test_malloc_response_is_8_bytes();
    run_test_free_request_is_8_bytes();
    run_test_memcpy_request_is_28_bytes();
    run_test_memset_request_is_20_bytes();
    run_test_register_module_request_is_8_bytes();
    run_test_register_module_response_is_8_bytes();
    run_test_register_function_request_is_20_bytes();
    run_test_register_function_response_size();
    run_test_launch_kernel_request_is_56_bytes();
    run_test_stream_create_request_is_4_bytes();
    run_test_stream_create_response_is_8_bytes();
    run_test_event_create_request_is_4_bytes();
    run_test_event_elapsed_time_response_is_4_bytes();
    run_test_set_memory_quota_request_is_8_bytes();
    run_test_usage_stats_response_size();
    run_test_report_usage_stats_size();

    /* Enums */
    run_test_connection_modes();
    run_test_agent_report_interval();
    run_test_report_cmd_id();

    /* Field offsets */
    run_test_header_field_offsets();

    /* Command IDs */
    run_test_command_id_ranges();

    /* Helpers */
    run_test_memcpy_h2d_payload_len();
    run_test_memcpy_h2d_payload_len_zero();

    /* Round-trips */
    run_test_header_roundtrip();
    run_test_launch_kernel_roundtrip();
    run_test_usage_stats_roundtrip();
    run_test_report_usage_stats_roundtrip();
    run_test_hello_with_conn_mode_roundtrip();

    /* Enums */
    run_test_memcpy_kinds();

    printf("\n%d/%d tests passed\n", g_tests_passed, g_tests_run);
    return g_tests_passed == g_tests_run ? 0 : 1;
}
