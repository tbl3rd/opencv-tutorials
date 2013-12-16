INSTALL := ../opencv/install
LIBS := \
-lopencv_core \
-lopencv_highgui \
-lopencv_imgproc \
#

CXXFLAGS := -g -O0
CXXFLAGS += -I$(INSTALL)/include
CXXFLAGS += -L$(INSTALL)/lib $(LIBS)

EXECUTABLE := loadAndDisplayImage
IMAGEFILE := ../resources/Twas_Ever_Thus500.jpg

main: $(EXECUTABLE)

help: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE)

test: main
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	./$(EXECUTABLE) $(IMAGEFILE)

clean:
	rm -rf $(EXECUTABLE) gray-image.jpg *.dSYM

debug: $(EXECUTABLE)
	DYLD_LIBRARY_PATH=$(INSTALL)/lib:$$DYLD_LIBRARY_PATH \
	lldb ./$(EXECUTABLE) -- $(IMAGEFILE)


.PHONY: main test clean debug
