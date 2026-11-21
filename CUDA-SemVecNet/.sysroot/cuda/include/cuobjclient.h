/*
 * SPDX-FileCopyrightText: Copyright (c) 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#ifndef _CUOBJCLIENT_H_
#define _CUOBJCLIENT_H_

#define OBJ_RDMA_V1 "CUOBJ"

#define CUOBJ_CLIENT_MAJOR_VERSION 1
#define CUOBJ_CLIENT_MINOR_VERSION 3

/* Version encoding: major * 1000 + minor * 10. */
#define CUOBJ_CLIENT_VERSION \
    (CUOBJ_CLIENT_MAJOR_VERSION * 1000 + \
     CUOBJ_CLIENT_MINOR_VERSION * 10)
/**
 * @brief Maximum memory registration size limit for RDMA operations (4GiB - 64K)
 * Aligned with CUFILE_MAX_RDMA_IO_SIZE to keep IO size within the 32-bit
 * descriptor field and avoid unnecessary chunking at the boundary.
 */
#define CUOBJ_MAX_MEMORY_REG_SIZE (4ULL * 1024 * 1024 * 1024 - 64 * 1024)

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>

#include "cufile.h"
#include "cuobjtelem.h"
#include "cuobjextrc_types.h"

/**
 * @brief CUOBJ-level RDMA error codes (negative of cuFile RDMA error codes)
 *
 * CU_OBJ_RDMA_RESET_TOKEN can be used in two ways to trigger a token reset:
 *   1. Returned from the GET/PUT callback to the library.
 *      After the reset, cuObjGet/cuObjPut returns:
 *        - -EIO if the reset SUCCEEDED (current I/O failed; application should retry).
 *        - CU_OBJ_RDMA_RESET_FAILED if the reset FAILED (application should
 *          synchronize all outstanding I/Os and call cuFileDriverClose() to recover).
 *   2. Passed as the err_code argument to cuMemObjPutRDMAToken().
 *      After the reset, cuMemObjPutRDMAToken returns:
 *        - CU_OBJ_SUCCESS if the reset SUCCEEDED.
 *        - CU_OBJ_RESET_FAIL if the reset FAILED.
 */
#define CU_OBJ_RDMA_RESET_TOKEN     (-(CU_FILE_RDMA_RESET_TOKEN))     /**< Pass to callback return or cuMemObjPutRDMAToken to trigger token reset */
#define CU_OBJ_RDMA_RESET_FAILED    (-(CU_FILE_RDMA_RESET_TOKEN_FAILED))    /**< QP reset failed */

/**
 * @file cuobjclient.h
 * @brief  cuObject client C++ APIs
 *
 * This file contains all the C++ APIs to perform GPUDirect Storage supported client IO operations for objects
 */


/**
 * Telemetry object
 */
extern std::shared_ptr<cuObjTelem> clientTelem;

/**
 * @brief cuObject error numbers.
 *
 * @note  These errors will be expanded in future
 *
 */

#ifndef CUOBJ_ERR_T_DEFINED
#define CUOBJ_ERR_T_DEFINED
typedef enum cuObjErr_enum {
	CU_OBJ_SUCCESS = 0,        /**< Operation successfully completed */
	CU_OBJ_FAIL = 1,           /**< Operation failed */
	CU_OBJ_RESET_FAIL = 2,     /**< Failed to reset QP */
	CU_OBJ_WRONG_PROTOCOL = 3, /**< API called on a client/server constructed with the wrong protocol */
	CU_OBJ_EXT_RC_NOT_REGISTERED = 4, /**< RC_V1 client/server has no External RC connection registered */
	CU_OBJ_INVALID_VALUE = 5,  /**< Invalid input value (e.g. size > CUOBJ_MAX_MEMORY_REG_SIZE) */
} cuObjErr_t;	
#endif /* CUOBJ_ERR_T_DEFINED */

/**
 * @brief cuObject RDMA descriptor protocol version
 *
 */
#ifndef CUOBJ_PROTO_T_DEFINED
#define CUOBJ_PROTO_T_DEFINED
typedef enum cuObjProto_enum {
	CUOBJ_PROTO_RDMA_DC_V1=1001, /**< RDMA DC (default) */
	CUOBJ_PROTO_RDMA_EXT_RC_V1=1002, /**< External RC transport; provider selected via cufile.json + registerExternalRCConnection */
	CUOBJ_PROTO_MAX
} cuObjProto_t;
#endif /* CUOBJ_PROTO_T_DEFINED */


/**
 * @brief cuObject Operation type
 *
 */
#ifndef CUOBJ_OPTYPE_T_DEFINED
#define CUOBJ_OPTYPE_T_DEFINED
typedef enum cuObjOpType_enum {
	CUOBJ_GET = 0, /**< GET operation */
	CUOBJ_PUT = 1, /**< PUT operation */
	CUOBJ_INVALID=9999
} cuObjOpType_t;
#endif /* CUOBJ_OPTYPE_T_DEFINED */

/**
 * @brief cuObject Operation callbacks
 * This struct specifies the callback interfaces used by cuObjClient class object during IO operations.
 * @note The callbacks can be called from a different thread than the caller thread. user must lock any shared resources
 * that can be used concurrently across multiple callers.
 */
typedef struct CUObjIOOps {
/**
 * @brief cuObject GET callback
 * @param handle cookie to the user context provided in the cuObjGet call. cuObjClient::getCtx(handle) should be called for getting the user context
 * @param ptr pointer to the start of the memory chunk
 * @param size size of the memory chunk being read.
 * @param offset starting object offset for this memory chunk.
 * @param cufileRDMAInfo_t Pointer to a RDMA memory descriptor string
 *
 * @return size of the data read on success, or a negative error code on failure.
 *         To trigger a token reset, the callback should return CU_OBJ_RDMA_RESET_TOKEN.
 *         The library will then attempt a token reset and cuObjGet will return:
 *         - -EIO: token reset SUCCEEDED; the current I/O has failed but the reset
 *           completed successfully. The application should retry the operation.
 *         - CU_OBJ_RDMA_RESET_FAILED: token reset FAILED; the application should
 *           synchronize all outstanding I/Os and call cuFileDriverClose() to
 *           recover.
 *
 * @note offset will be set to zero for cases where the MaxReqCallbackSize is equal to or greater the cuObjectGet call size
 * @note size will be set to total requested size n cuObjectGet for cases where the MaxReqCallbackSize is equal to or greater the cuObjectGet call size
 *
 *
 * @see cuObjClient::cuObjGet
 */

      ssize_t (*get) (const void *handle, char *ptr, size_t size, loff_t offset, const cufileRDMAInfo_t*);
/**
 * @brief cuObject PUT callback
 * @param handle to the user context provided in the cuObjPut call. cuObjClient::getCtx(handle) should be called for getting the user context
 * @param ptr pointer to the start of the memory chunk
 * @param size size of the memory chunk being written
 * @param offset starting object offset for this memory chunk.
 * @param cufileRDMAInfo_t Pointer to a RDMA memory descriptor string
 *
 * @return size of the data written on success, or a negative error code on failure.
 *         To trigger a token reset, the callback should return CU_OBJ_RDMA_RESET_TOKEN.
 *         The library will then attempt a token reset and cuObjPut will return:
 *         - -EIO: token reset SUCCEEDED; the current I/O has failed but the reset
 *           completed successfully. The application should retry the operation.
 *         - CU_OBJ_RDMA_RESET_FAILED: token reset FAILED; the application should
 *           synchronize all outstanding I/Os and call cuFileDriverClose() to
 *           recover.
 *
 * @note offset will be set to zero for cases where the MaxReqCallbackSize is equal to or greater the cuObjectPut call size
 * @note size will be set to total requested size n cuObjectGet for cases where the MaxReqCallbackSize is equal to or greater the cuObjectPut call size
 *
 *
 * @see cuObjClient::cuObjPut
 */

      ssize_t (*put) (const void *handle, const char *ptr, size_t size, loff_t offset, const cufileRDMAInfo_t*);
}CUObjOps_t;

/**
 * @brief cuObject memory type
 *
 */
typedef enum cuObjMemoryType_enum {
	CUOBJ_MEMORY_SYSTEM = 0,
	CUOBJ_MEMORY_CUDA_MANAGED = 1,
	CUOBJ_MEMORY_CUDA_DEVICE = 2,
	CUOBJ_MEMORY_UNKNOWN = 3,
	CUOBJ_MEMORY_INVALID = 4
} cuObjMemoryType_t;

extern "C" {

/**
 * @brief Client-side registration parameters for an External RC connection.
 *
 * `device_name` is caller-owned and must outlive the registration.
 */
struct cuObjClientRCRegistration {
    const char *device_name;                            /**< RDMA device name (e.g. "mlx5_0") — REQUIRED */
    struct cuobj_ext_rc_provider_config provider_config; /**< Provider and optional auth token */
    uint32_t logical_id;                                /**< Shared identifier with the server — REQUIRED, non-zero */
    union ibv_gid local_gid;                            /**< Local GID for this device */
    uint16_t      local_lid;                            /**< Local LID (0 for RoCE) */
    uint8_t       is_global;                            /**< 1 = GID-routed, 0 = LID only */
    cuObjExternalRCErrorCallback error_callback;        /**< Optional error callback — may be NULL */
    void                        *callback_context;     /**< Context passed to error_callback — may be NULL */
};

} /* extern "C" */

/**
 * @brief cuObjClient class.
 *
 * cuObjClient Provides client side APIs to prepare PUT/GET operations for out-of-band RDMA IO operations.
 * The user of this object is expected to implement a set of callback interfaces specified in CUObjIOOps.
 * Once the client object is created, user is expected to optionally register the memory for RDMA and
 * perform PUT and GET operations on this client object.
 * The cuObjClient will validate the memory and prepare a memory region for RDMA transer.
 * The cuObjClient will call one or more callback operations with relevant RDMA information.
 * The user is expected to relay the RDMA information and other data to cuObjServer using standard control path.
 * After the IO completion, the callback should return the total data read or written in this context.
 * The client object may perform additional operations before the cuObjectPut/cuObjectGet operation finishes.
 */

class cuObjClient {
	public:

                /**
                * @brief constructor for cuObjClient class.
                * @param ops callback reference to CUObjIOOps
                * @param proto RDMA descriptor protocol used for this client. defaults to CUOBJ_PROTO_RDMA_DC_V1
                */
		cuObjClient(CUObjOps_t& ops, cuObjProto_t proto=CUOBJ_PROTO_RDMA_DC_V1);
		cuObjClient(const cuObjClient& other) = delete;
		cuObjClient& operator=(const cuObjClient& other) = delete;
		~cuObjClient();

                /**
                * @brief Acquire a RDMA memory descriptor for the user memory
                * @param ptr start address of user memory
                * @param size size of memory that needs pinning starting from the start address of user memory
                * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure
                * @note Individual RDMA tokens and callback requests are limited to
                *       CUOBJ_MAX_MEMORY_REG_SIZE (4GiB - 64K).
                */
		cuObjErr_t cuMemObjGetDescriptor(void *ptr, size_t size);
                /**
                * @brief Get a RDMA memory descriptor for the user memory
                * @param ptr start address of user memory
                * @return max size of the callback for this memory pointer
                * @note: The size can be smaller than allocated memory if the memory is not registered or
                * the underlying RDMA subsystem does not allow for pinning/transfer of the entire memory in a single callback
                */
		ssize_t cuMemObjGetMaxRequestCallbackSize(void *ptr);

                /**
                * @brief release the RDMA memory descriptor for the user memory
                * @param ptr start address of user memory used during cuMemObjGetDescriptor
                * @return error status if the memory cannot be unregistered
                */

		cuObjErr_t cuMemObjPutDescriptor(void *ptr);

                /**
                * @brief Get RDMA descriptor string for a registered memory buffer
                * 
                * Gets the RDMA descriptor string and increments QP refcount for token reset support.
                * The descriptor string contains all RDMA information needed for remote access.
                *
                * @param ptr start address of user memory that was registered with cuMemObjGetDescriptor
                * @param size size of the memory region for which to generate the descriptor string
                * @param buffer_offset offset from the base address to start the descriptor region
                * @param operation operation type (CUOBJ_GET, CUOBJ_PUT)
                * @param desc_str_out pointer to store the allocated descriptor string (caller must free using cuMemObjPutRDMAToken)
                * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure
                * @note This function allocates memory for the descriptor string.
                * @note Caller MUST call cuMemObjPutRDMAToken to free it.
                * @note The buffer must be registered with cuMemObjGetDescriptor and be RDMA capable.
                *
                * @warning The application MUST ensure the following throughout the lifetime of the
                *          returned RDMA token (i.e., between cuMemObjGetRDMAToken and cuMemObjPutRDMAToken):
                *          - The buffer registration (from cuMemObjGetDescriptor) must remain valid.
                *            Calling cuMemObjPutDescriptor while an RDMA token is held is undefined behavior.
                *          - cuFileDriverClose() must NOT be called while any RDMA token is outstanding.
                *          Violating these constraints may lead to use-after-free or undefined behavior.
                */
		cuObjErr_t cuMemObjGetRDMAToken(void *ptr, size_t size, size_t buffer_offset, 
		                                 cuObjOpType_t operation, char **desc_str_out);

                /**
                * @brief Free RDMA descriptor string and decrement QP refcount
                * 
                * Frees the descriptor string allocated by cuMemObjGetRDMAToken and decrements the
                * QP refcount. The device is found by matching GID/LID from the descriptor string.
                *
                * @param desc_str descriptor string to free (from cuMemObjGetRDMAToken)
                * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure
                *
                * @warning See cuMemObjGetRDMAToken for lifetime requirements. The buffer registration
                *          and driver must remain valid until this call completes.
                */
		cuObjErr_t cuMemObjPutRDMAToken(char *desc_str);

                /**
                * @brief Free RDMA descriptor string with optional token reset
                *
                * Frees the descriptor string and decrements QP refcount. If CU_OBJ_RDMA_RESET_TOKEN
                * is passed as err_code, also triggers a token reset.
                *
                * @param desc_str descriptor string to free (from cuMemObjGetRDMAToken)
                * @param err_code error code for token management
                *        - 0: just decrement refcount and free
                *        - CU_OBJ_RDMA_RESET_TOKEN: decrement refcount, then trigger token reset (BLOCKING)
                * @return CU_OBJ_SUCCESS on success, CU_OBJ_FAIL on failure, CU_OBJ_RESET_FAIL on reset failure
                * @note CU_OBJ_RDMA_RESET_TOKEN is a BLOCKING call that waits for pending I/O to drain.
                * @warning CU_OBJ_RDMA_RESET_TOKEN should only be used in recovery scenarios.
                *
                * @warning See cuMemObjGetRDMAToken for lifetime requirements. The buffer registration
                *          and driver must remain valid until this call completes.
                */
		cuObjErr_t cuMemObjPutRDMAToken(char *desc_str, int err_code);

                /**
                * @brief Get the user context provided in the cuObjGet and cuObjPut from handle in the callback
                * @param handle pointer to the handle from the callback
                * @return void pointer to the user context.
                */

		static void* getCtx(const void *handle);

                /**
                * @brief API to perform GET operation using cuObject
                * @param ctx pointer to a user control context, used in GET callback
                * @param ptr pointer to a user memory
                * @param size size of the GET operation
                * @param offset currently set to 0. reserved for future use
                * @param buf_offset currently set to 0. reserved for future use
                * @return size of data read on success, or a negative error code:
                *         - -EINVAL: input parameters are invalid.
                *         - -EIO: The GET callback returned CU_OBJ_RDMA_RESET_TOKEN and
                *           the QP reset SUCCEEDED. The current I/O has failed but a new
                *           QP is now active. The application should retry the operation.
                *         - CU_OBJ_RDMA_RESET_FAILED: The GET callback returned
                *           CU_OBJ_RDMA_RESET_TOKEN but the QP reset FAILED (e.g., drain
                *           timeout). The application should synchronize all outstanding
                *           I/Os and call cuFileDriverClose() to recover.
                *         - Other negative values indicate standard I/O errors.
                */

		ssize_t cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset=0, loff_t buf_offset=0);
                /**
                * @brief API to perform PUT operation using cuObject
                * @param ctx pointer to a user control context, used in PUT callback
                * @param ptr pointer to a user memory
                * @param size size of the PUT operation
                * @param offset currently set to 0. reserved for future use
                * @param buf_offset currently set to 0. reserved for future use
                * @return size of data written on success, or a negative error code:
                *         - -EINVAL: input parameters are invalid.
                *         - -EIO: The PUT callback returned CU_OBJ_RDMA_RESET_TOKEN and
                *           the QP reset SUCCEEDED. The current I/O has failed but a new
                *           QP is now active. The application should retry the operation.
                *         - CU_OBJ_RDMA_RESET_FAILED: The PUT callback returned
                *           CU_OBJ_RDMA_RESET_TOKEN but the QP reset FAILED (e.g., drain
                *           timeout). The application should synchronize all outstanding
                *           I/Os and call cuFileDriverClose() to recover.
                *         - Other negative values indicate standard I/O errors.
                */

		ssize_t cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset=0, loff_t buf_offset=0);

		/**
		 * @brief Register an external RC connection on this client.
		 * @param reg  Registration parameters; `logical_id` must be non-zero.
		 * @return CU_OBJ_SUCCESS on success; CU_OBJ_FAIL on duplicate or invalid input.
		 * @note Thread-safe.
		 */
		cuObjErr_t registerExternalRCConnection(const cuObjClientRCRegistration& reg);

		/**
		 * @brief Unregister one or all external RC connections.
		 * @param logical_id  Specific logical_id to drop, or -1 to drop all.
		 * @param timeout_ms  Max time to wait for in-flight I/O to drain.
		 */
		cuObjErr_t unregisterExternalRCConnection(int logical_id = -1,
		                                          uint32_t timeout_ms = 5000);

		/**
		 * @brief Enumerate the client-side RDMA Protection Domains.
		 * @param list         Caller-allocated array of cuObjRDMAPDInfo.
		 * @param max_entries  Capacity of `list`.
		 * @return >=0 entry count on success, -1 on error.
		 */
		static int getRDMAPDList(struct cuObjRDMAPDInfo* list, int max_entries);

                /**
                * @brief check if the client is connected
                */
		bool isConnected(void);
                /**
                * @brief setup telemetry output stream. must call shutdownTelemetry() before closing the os
                * @warning the os must be valid until all the cuObjClient objects are destroyed
                */
                static void setupTelemetry(bool use_OTEL, std::ostream *os);

                /**
                * @brief set telemetry stream logging level and op logging flags
                */
                static void setTelemFlags(unsigned log_flags);
                static void setTelemFlags(unsigned log_flags, unsigned log_op_flags);

                /**
                * @brief setup telemetry op update frequency
                */
                static void setOpFreq(unsigned freq_get, unsigned freq_put);

                /**
                * @brief shutdown telemetry. reset the telemetry to default output stream
                * @note the telemetry will be closed when all the cuObjClient objects are destroyed
                */
                static void shutdownTelemetry();

                 /**
                  * @brief Get the memory type of a given pointer
                  * @param ptr pointer to the memory
                  * @return memory type
                  */
                 static cuObjMemoryType_t getMemoryType(const void* ptr);

        private:
		struct Impl;
		Impl *_impl;
#if UINTPTR_MAX == UINT64_MAX
		enum { CUOBJCLIENT_ABI_RESERVED_PTRS = 8 };
#else
#error "cuObjClient SO1 ABI layout is supported only on 64-bit platforms"
#endif
		void *_reserved[CUOBJCLIENT_ABI_RESERVED_PTRS];
};

#endif
