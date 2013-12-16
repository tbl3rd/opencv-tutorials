INSTALL := ../opencv/install
LIBS := \
-lopencv_core \
-lopencv_highgui \
-lopencv_imgproc \
#

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := compareHistogram
IMAGEFILE := \
	../resources/hand_sample1.jpg \
	../resources/hand_sample2.jpg \
	../resources/hand_sample3.jpg \
	#

main: $(EXECUTABLE)

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH ./$(EXECUTABLE)

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(IMAGEFILE)

clean:
	rm -rf $(EXECUTABLE) *.dSYM

debug: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- $(IMAGEFILE)

.PHONY: main help test clean debug
