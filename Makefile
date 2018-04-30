
#############################################################################
# Constants
#############################################################################

TARGET    := pnlp

# Source directories
SRC_DIR   := src
# Include directories
INC_DIR   := include
INC_DIR   += ../rapidjson/include

# Source files
SRC_FILES := $(foreach i,$(SRC_DIR),$(wildcard $i/*.cpp))
# Object files
OBJ_FILES := $(subst .cpp,.o,$(filter %.cpp,$(SRC_FILES)))
OBJ_FILES := $(addprefix obj/,$(notdir $(OBJ_FILES)))
# Dependency files
DEP_FILES := $(subst .o,.d,$(OBJ_FILES))
DEP_FILES := $(addprefix dep/,$(notdir $(DEP_FILES)))

# Flags
CXXFLAGS  := #-Wall -Wno-unused-but-set-variable -Wno-unused-result -Wno-unused-variable
CXXFLAGS  += -O3 #-std=c++11
CXXFLAGS  += $(addprefix -I,$(INC_DIR))

LDFLAGS   :=

# Utilities
SHELL     := /bin/sh
CXX       := g++
SED       := sed
RM        := rm -f
MAKE      := make
CD        := cd
CP        := cp -f

VPATH    := $(SRC_DIR)

ifneq "$(MAKECMDGOALS)" "clean"
-include $(DEP_FILES)
endif

#############################################################################
# Rules
#############################################################################

.DEFAULT_GOAL := default

.PHONY: default
default: $(OBJ_FILES)
	@echo "  Linking $(TARGET)"
	@$(CXX) $(OBJ_FILES) $(LDFLAGS) -o $(TARGET)

.PHONY: clean
clean:
	@$(RM) $(TARGET)
	@$(RM) $(OBJ_FILES)
	@$(RM) $(DEP_FILES)

obj/%.o: %.cpp
	@echo "  CXX $(notdir $<)"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

dep/%.d: %.cpp
	@$(CXX) $(CXXFLAGS) -MM $< > $@.$$$$; \
	$(SED) 's,\($(notdir $*)\)\.o[ :]*,obj/\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

