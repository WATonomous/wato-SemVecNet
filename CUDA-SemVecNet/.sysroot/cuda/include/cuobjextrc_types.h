/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef CUOBJEXTRC_TYPES_H
#define CUOBJEXTRC_TYPES_H

/**
 * @file cuobjextrc_types.h
 * @brief Shared public types for External RC connection management.
 */

#include <stdint.h>
#include <stdbool.h>
#include <infiniband/verbs.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief External RC provider identifier.
 */
typedef enum cuobj_ext_rc_provider_enum {
    CUOBJ_EXT_RC_PROVIDER_AZURE = 1,
} cuobj_ext_rc_provider_t;

/**
 * @brief Per-connection provider configuration.
 *
 * `auth_token` is caller-owned and must outlive the registration.
 */
struct cuobj_ext_rc_provider_config {
    cuobj_ext_rc_provider_t provider;
    bool                    has_auth;
    const char             *auth_token;
};

typedef enum cuObjExternalRCState_enum {
    CUOBJ_EXT_RC_STATE_CONNECTED = 0,
    CUOBJ_EXT_RC_STATE_ERROR     = 1,
    CUOBJ_EXT_RC_STATE_DRAINING  = 2,
    CUOBJ_EXT_RC_STATE_DISABLED  = 3,
} cuObjExternalRCState_t;

typedef enum cuObjExternalRCError_enum {
    CUOBJ_EXT_RC_ERR_NONE             = 0,
    CUOBJ_EXT_RC_ERR_QP_ERROR         = 1,
    CUOBJ_EXT_RC_ERR_COMPLETION_ERROR = 2,
    CUOBJ_EXT_RC_ERR_TIMEOUT          = 3,
    CUOBJ_EXT_RC_ERR_REMOTE_ACCESS    = 4,
    CUOBJ_EXT_RC_ERR_TRANSPORT        = 5,
    CUOBJ_EXT_RC_ERR_UNKNOWN          = 99,
} cuObjExternalRCError_t;

struct cuObjExternalRCStats {
    uint64_t rdma_writes;
    uint64_t rdma_reads;
    uint64_t rdma_write_bytes;
    uint64_t rdma_read_bytes;

    uint64_t completion_errors;
    uint64_t post_send_errors;
    uint64_t total_errors;

    uint64_t total_write_time_us;
    uint64_t total_read_time_us;
    uint64_t min_write_latency_us;
    uint64_t max_write_latency_us;
    uint64_t min_read_latency_us;
    uint64_t max_read_latency_us;

    uint64_t state_changes;
    uint64_t time_in_error_us;
};

struct cuObjExternalRCErrorEvent {
    uint32_t               logical_id;
    uint16_t               channel;
    cuObjExternalRCError_t error;
    cuObjExternalRCState_t old_state;
    cuObjExternalRCState_t new_state;
    enum ibv_wc_status     wc_status;
    uint64_t               wr_id;
    const char            *error_message;

    union ibv_gid client_gid;
    uint16_t      client_lid;
};

typedef void (*cuObjExternalRCErrorCallback)(
    const struct cuObjExternalRCErrorEvent *event,
    void *user_context
);

struct cuObjExternalRCInfo {
    uint32_t logical_id;
    uint16_t channel;
    char     device_name[64];

    union ibv_gid client_gid;
    uint16_t      client_lid;
    uint8_t       is_global;

    uint32_t          qp_num;
    enum ibv_qp_state qp_state;

    cuObjExternalRCState_t      state;
    struct cuObjExternalRCStats stats;
    int                         pending_ops;
};

#define CUOBJ_MAX_RDMA_PD_ENTRIES 16
#define CUOBJ_MAX_GIDS_PER_PD     16

struct cuObjRDMAPDGIDInfo {
    union ibv_gid gid;
    uint16_t      lid;
    int           gid_index;
    char          ip_addr[64];
};

/**
 * @brief RDMA Protection Domain information returned by getRDMAPDList().
 */
struct cuObjRDMAPDInfo {
    struct ibv_pd      *pd;
    struct ibv_context *context;
    char                device_name[16];
    int                 ib_port;

    int                       num_gids;
    struct cuObjRDMAPDGIDInfo gids[CUOBJ_MAX_GIDS_PER_PD];
};

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUOBJEXTRC_TYPES_H */
