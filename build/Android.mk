LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE := neon_mat_mul

LOCAL_CFLAGS += -D__DEBUG__

PROJECT_SRC = $(LOCAL_PATH)/../src
PROJECT_INC = $(LOCAL_PATH)/../inc
PROJECT_UTIL = $(LOCAL_PATH)/../util/inc

LOCAL_C_INCLUDES += $(PROJECT_INC) \
					$(PROJECT_UTIL) \

LOCAL_SRC_FILES += $(PROJECT_SRC)/neon_mat_main.c \
				   $(PROJECT_SRC)/neon_mat_mul.c

LOCAL_LDLIBS := -lm -llog
LOCAL_LDFLAGS := -nodefaultlibs -lc -lm -ldl

include $(BUILD_EXECUTABLE)
