
#############################################################################
# Constants
#############################################################################

TARGET       := PH_TARGET

# Source directories
SRC_DIR      := src eval/src
# Include directories
INC_DIR      := include eval/include
INC_DIR      += ../../../ipopt/cur/include/coin

# Source files
SRC_FILES    := $(foreach i,$(SRC_DIR),$(wildcard $i/*.cpp))
SRC_FILES    += $(foreach i,$(SRC_DIR),$(wildcard $i/*.cu))
# Object files
OBJ_FILES    := $(subst .cpp,.o,$(filter %.cpp,$(SRC_FILES)))
OBJ_FILES    := $(addprefix obj/,$(notdir $(OBJ_FILES)))
CU_OBJ_FILES := $(subst .cu,.o,$(filter %.cu,$(SRC_FILES)))
CU_OBJ_FILES := $(addprefix obj/,$(notdir $(CU_OBJ_FILES)))
# Dependency files
DEP_FILES    := $(subst .o,.d,$(OBJ_FILES))
DEP_FILES    += $(subst .o,.d,$(CU_OBJ_FILES))
DEP_FILES    := $(addprefix dep/,$(notdir $(DEP_FILES)))

# Flags
CXXFLAGS     := #-Wall -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable
CXXFLAGS     += -O3 #-std=c++11
CXXFLAGS     += $(addprefix -I,$(INC_DIR))

NVCCFLAGS    := $(addprefix -I,$(INC_DIR)) --gpu-architecture=sm_35 --disable-warnings
#NVCCFLAGS += -Xptxas --disable-optimizer-constants
#NVCCFLAGS += --ptxas-options=-v

LDFLAGS      := -L${CUDA_HOME}/lib64 -lcudadevrt -lcudart
#LDFLAGS      += -L${MKL_HOME}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lmkl_rt -lpthread -lm -ldl
LDFLAGS      += -L${COINHSL_HOME}/lib -lcoinhsl -L${IPOPT_HOME}/lib -lipopt

# Utilities
SHELL     := /bin/sh
NVCC      := nvcc
CXX       := g++
SED       := sed
RM        := rm -f
MAKE      := make
CD        := cd
CP        := cp -f

VPATH     := $(SRC_DIR)

ifneq "$(MAKECMDGOALS)" "clean"
-include $(DEP_FILES)
endif

#############################################################################
# Rules
#############################################################################

.DEFAULT_GOAL := default

.PHONY: default
default: $(OBJ_FILES) $(CU_OBJ_FILES)
	@echo "  NVCC linking $(TARGET)"
	@$(NVCC) $(NVCCFLAGS) $(OBJ_FILES) $(CU_OBJ_FILES) $(LDFLAGS) -o $(TARGET)

.PHONY: clean
clean:
	@$(RM) $(TARGET)
	@$(RM) $(OBJ_FILES)
	@$(RM) $(CU_OBJ_FILES)
	@$(RM) $(DEP_FILES)

obj/%.o: %.cpp
	@echo "  CXX  $(notdir $<)"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

dep/%.d: %.cpp
	@$(CXX) $(CXXFLAGS) -M $< > $@.$$$$; \
	$(SED) 's,\($(notdir $*)\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

obj/%.o: %.cu
	@echo "  NVCC $(notdir $<)"
	@$(NVCC) $(NVCCFLAGS) --device-c $< -o $@

dep/%.d: %.cu
	@$(NVCC) $(NVCCFLAGS) -M $< > $@.$$$$; \
	$(SED) 's,\($(notdir $*)\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

